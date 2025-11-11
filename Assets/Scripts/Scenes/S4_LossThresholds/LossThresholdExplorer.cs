using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System;
using System.Collections.Generic;
using System.Globalization;

public class LossThresholdExplorer : MonoBehaviour
{
    private const string SCENE_ID = "S4_LossThresholds";
    private const float SUCCESS_F_THRESHOLD = 0.85f;

    [Header("Data & World")]
    public Dataset2D dataset;
    public Transform pointsParent;
    public GameObject pointPrefab;

    [Header("UI")]
    public Button btnStep;
    public Toggle tglAuto, tglMinibatch;
    public Slider sldBatch, sldLR;
    public TMP_Dropdown drpLossType;    // 0=BCE, 1=MSE
    public Slider sldThreshold;         // 0..1
    public TMP_Text txtMetrics;
    public Button btnReset, btnShuffle;

    [Header("Panels")]
    public ProbRailPanel probRail;
    public ROCPanel rocPanel;
    public LossComparePanel lossPanel;

    [Header("Point Style")]
    [Range(0.02f, 0.30f)] public float dotScale = 0.08f;
    public Color blueColor = new Color(0.45f, 0.62f, 0.94f, 0.85f);
    public Color redColor = new Color(0.94f, 0.45f, 0.45f, 0.85f);

    // internals
    private MLP mlp;
    private float[,] X, Y;
    private readonly List<GameObject> pointGOs = new();
    private System.Random rnd = new System.Random(7);
    private bool auto = false;
    private float timer = 0f;
    private const float dt = 0.05f;
    private int stepCount = 0;
    private float elapsed = 0f;

    // Gamification
    private ObjectiveTracker _tracker;

    private void Start()
    {
        // --- Log scene start ---
        EventLogger.Instance?.LogEvent("SceneStart", key: SCENE_ID);
        EvalLogger.Instance?.Info("SceneStart_S4", new Dictionary<string, object>
        {
            { "sceneId", SCENE_ID }
        });

        // dataset clone & init
        dataset = dataset
            ? ScriptableObject.Instantiate(dataset)
            : ScriptableObject.CreateInstance<Dataset2D>();

        if (dataset.points == null || dataset.points.Length == 0)
            dataset.GenerateBlobs();

        dataset.ReseedAndGenerateClean();
        X = dataset.XMatrix();
        Y = dataset.YMatrix();

        // model init
        mlp = new MLP(2, 3, 1, seed: UnityEngine.Random.Range(1, 1_000_000))
        {
            lossType = LossType.BCE,
            lr = sldLR.value
        };

        SpawnPoints();

        // tracker
        _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();

        // --- UI hooks ---

        btnStep.onClick.AddListener(() =>
        {
            Step();
            _tracker?.ReportAction("step_train");
            EventLogger.Instance?.LogEvent("Action", key: "step_train");
            EvalLogger.Instance?.ActionEvent("S4_StepManual", new Dictionary<string, object>
            {
                { "step", stepCount }
            });
        });

        tglAuto.onValueChanged.AddListener(v =>
        {
            auto = v;
            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "auto_mode",
                value: v ? "on" : "off"
            );
            EvalLogger.Instance?.ActionEvent("S4_ToggleAuto", new Dictionary<string, object>
            {
                { "isOn", v }
            });
        });

        tglMinibatch.onValueChanged.AddListener(v =>
        {
            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "minibatch_enabled",
                value: v ? "true" : "false"
            );
            EvalLogger.Instance?.ActionEvent("S4_ToggleMinibatch", new Dictionary<string, object>
            {
                { "isOn", v }
            });
        });

        sldBatch.onValueChanged.AddListener(v =>
        {
            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "batch_size",
                value: ((int)v).ToString()
            );
            EvalLogger.Instance?.ActionEvent("S4_ChangeBatchSize", new Dictionary<string, object>
            {
                { "batchSize", (int)v }
            });
        });

        sldLR.onValueChanged.AddListener(v =>
        {
            mlp.lr = v;
            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "learning_rate",
                value: v.ToString("F4", CultureInfo.InvariantCulture)
            );
            EvalLogger.Instance?.ActionEvent("S4_ChangeLR", new Dictionary<string, object>
            {
                { "lr", v }
            });
        });

        drpLossType.onValueChanged.AddListener(OnLossChanged);
        sldThreshold.onValueChanged.AddListener(OnThresholdSliderChanged);

        btnReset.onClick.AddListener(() =>
        {
            RecordRunSummary("Reset");
            ResetAll();
        });

        btnShuffle.onClick.AddListener(() =>
        {
            RecordRunSummary("Shuffle");
            ShuffleData();
        });

        // initial draw & metrics
        RefreshAll();
        ReportMetricsToTrackerFromCurrent();
        LogFaithfulnessSnapshot();
    }

    private void Update()
    {
        elapsed += Time.deltaTime;

        if (!auto) return;

        timer += Time.deltaTime;
        if (timer >= dt)
        {
            timer = 0f;
            Step();
            _tracker?.ReportAction("step_train_auto");
            EventLogger.Instance?.LogEvent("Action", key: "step_train_auto");
            EvalLogger.Instance?.ActionEvent("S4_StepAuto", new Dictionary<string, object>
            {
                { "step", stepCount }
            });
        }
    }

    private void OnDestroy()
    {
        if (stepCount > 0)
        {
            RecordRunSummary("SceneExit");
        }
    }

    // ---------------- UI / Core Logic ----------------

    private void OnLossChanged(int idx)
    {
        mlp.lossType = idx == 0 ? LossType.BCE : LossType.MSE;

        EventLogger.Instance?.LogEvent(
            eventType: "ParamChange",
            key: "loss_type",
            value: mlp.lossType.ToString()
        );
        EvalLogger.Instance?.ActionEvent("S4_ChangeLossType", new Dictionary<string, object>
        {
            { "lossType", mlp.lossType.ToString() }
        });

        RefreshAll();
        ReportMetricsToTrackerFromCurrent();
        LogFaithfulnessSnapshot();
    }

    private void OnThresholdSliderChanged(float value)
    {
        RefreshPanelsOnly();
        ReportThresholdChange(value);
        ReportMetricsToTrackerFromCurrent();
        LogFaithfulnessSnapshot();
    }

    private void Step()
    {
        int N = dataset.count;

        if (tglMinibatch && tglMinibatch.isOn)
        {
            int bs = Mathf.Clamp((int)sldBatch.value, 1, N);
            var idx = SampleBatch(bs, N);
            var (Xb, Yb) = GatherBatch(idx);
            mlp.Forward(Xb, Yb);
            mlp.StepSGD(bs);
        }
        else
        {
            mlp.Forward(X, Y);
            mlp.StepSGD(N);
        }

        stepCount++;

        RefreshAll();
        ReportMetricsToTrackerFromCurrent();
        LogFaithfulnessSnapshot();

        EventLogger.Instance?.LogEvent(
            eventType: "StepTrain",
            key: "step",
            value: stepCount.ToString()
        );
        EvalLogger.Instance?.Metric("S4_StepTrain", new Dictionary<string, object>
        {
            { "step", stepCount }
        });
    }

    // ---------------- Rendering & Panels ----------------

    private void RefreshAll()
    {
        var (loss, P) = mlp.Forward(X, Y);
        float thr = sldThreshold.value;

        var (TP, FP, TN, FN, prec, rec, f1, acc) = Metrics(P, Y, thr);

        if (txtMetrics)
        {
            txtMetrics.text =
                $"LossType: {mlp.lossType} | Loss: {loss:F4}\n" +
                $"Thr: {thr:F2} | TP:{TP} FP:{FP} TN:{TN} FN:{FN}\n" +
                $"Precision:{prec:0.000}  Recall:{rec:0.000}  F1:{f1:0.000}  Acc:{acc:0.000}";
        }

        probRail?.Redraw(P, Y, thr);
        rocPanel?.Redraw(P, Y, thr);
        lossPanel?.Redraw(P, Y);

        UpdatePointStyles(P);
    }

    private void RefreshPanelsOnly()
    {
        var (_, P) = mlp.Forward(X, Y);
        float thr = sldThreshold.value;

        var (TP, FP, TN, FN, prec, rec, f1, acc) = Metrics(P, Y, thr);

        if (txtMetrics)
        {
            txtMetrics.text =
                $"LossType: {mlp.lossType} | Loss: {mlp.lossType} (recalc)\n" +
                $"Thr: {thr:F2} | TP:{TP} FP:{FP} TN:{TN} FN:{FN}\n" +
                $"Precision:{prec:0.000}  Recall:{rec:0.000}  F1:{f1:0.000}  Acc:{acc:0.000}";
        }

        probRail?.Redraw(P, Y, thr);
        rocPanel?.Redraw(P, Y, thr);
        lossPanel?.Redraw(P, Y);

        UpdatePointStyles(P);
    }

    private void ResetAll()
    {
        mlp = new MLP(2, 3, 1, seed: UnityEngine.Random.Range(1, 1_000_000))
        {
            lossType = (drpLossType.value == 0 ? LossType.BCE : LossType.MSE),
            lr = sldLR.value
        };

        stepCount = 0;
        elapsed = 0f;

        RefreshAll();
        ReportMetricsToTrackerFromCurrent();
        LogFaithfulnessSnapshot();

        EventLogger.Instance?.LogEvent("ResetPressed");
        EvalLogger.Instance?.ActionEvent("S4_Reset", null);
    }

    private void ShuffleData()
    {
        dataset.ReseedAndGenerateClean();
        X = dataset.XMatrix();
        Y = dataset.YMatrix();
        SpawnPoints();

        stepCount = 0;
        elapsed = 0f;

        RefreshAll();
        ReportMetricsToTrackerFromCurrent();
        LogFaithfulnessSnapshot();

        EventLogger.Instance?.LogEvent("ShuffleData");
        EvalLogger.Instance?.ActionEvent("S4_ShuffleData", null);
    }

    private void SpawnPoints()
    {
        for (int i = pointsParent.childCount - 1; i >= 0; i--)
            Destroy(pointsParent.GetChild(i).gameObject);
        pointGOs.Clear();

        for (int i = 0; i < dataset.points.Length; i++)
        {
            var go = Instantiate(pointPrefab, dataset.points[i], Quaternion.identity, pointsParent);
            var sr = go.GetComponent<SpriteRenderer>();
            sr.color = dataset.labels[i] > 0.5f ? redColor : blueColor;
            sr.sortingOrder = 10;
            go.transform.localScale = Vector3.one * dotScale;
            pointGOs.Add(go);
        }
    }

    private void UpdatePointStyles(float[,] P)
    {
        for (int i = 0; i < pointGOs.Count; i++)
        {
            var sr = pointGOs[i].GetComponent<SpriteRenderer>();
            bool isRed = dataset.labels[i] > 0.5f;
            Color baseCol = isRed ? redColor : blueColor;

            float conf = Mathf.Clamp01(Mathf.Abs(P[i, 0] - 0.5f) * 2f);
            float a = Mathf.Lerp(0.25f, baseCol.a, conf);
            baseCol.a = a;

            sr.color = baseCol;
        }
    }

    // ---------------- Metrics (Confusion, F1, Acc) ----------------

    private (int, int, int, int, float, float, float, float) Metrics(float[,] P, float[,] Y, float thr)
    {
        int N = P.GetLength(0);
        int TP = 0, FP = 0, TN = 0, FN = 0;
        for (int i = 0; i < N; i++)
        {
            int y = Y[i, 0] > 0.5f ? 1 : 0;
            int h = P[i, 0] >= thr ? 1 : 0;
            if (h == 1 && y == 1) TP++;
            else if (h == 1 && y == 0) FP++;
            else if (h == 0 && y == 0) TN++;
            else FN++;
        }

        float prec = TP + FP == 0 ? 0f : TP / (float)(TP + FP);
        float rec = TP + FN == 0 ? 0f : TP / (float)(TP + FN);
        float f1 = (prec + rec) == 0 ? 0f : 2f * prec * rec / (prec + rec);
        float acc = (TP + TN) / (float)Mathf.Max(1, (TP + TN + FP + FN));
        return (TP, FP, TN, FN, prec, rec, f1, acc);
    }

    // minibatch helpers
    private int[] SampleBatch(int bs, int total)
    {
        var set = new HashSet<int>();
        while (set.Count < bs) set.Add(rnd.Next(total));
        var arr = new int[bs];
        int k = 0;
        foreach (var i in set) arr[k++] = i;
        return arr;
    }

    private (float[,], float[,]) GatherBatch(int[] idx)
    {
        var Xb = new float[idx.Length, 2];
        var Yb = new float[idx.Length, 1];
        for (int r = 0; r < idx.Length; r++)
        {
            int i = idx[r];
            Xb[r, 0] = dataset.points[i].x;
            Xb[r, 1] = dataset.points[i].y;
            Yb[r, 0] = dataset.labels[i];
        }
        return (Xb, Yb);
    }

    // ---------- Gamification glue ----------

    private void ReportThresholdChange(float value)
    {
        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();

        _tracker?.ReportAction("threshold_change");

        EventLogger.Instance?.LogEvent(
            eventType: "ParamChange",
            key: "threshold",
            value: value.ToString("F3", CultureInfo.InvariantCulture)
        );
    }

    private void ReportMetricsToTrackerFromCurrent()
    {
        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();
        if (_tracker == null || mlp == null || X == null || Y == null) return;

        var (_, P) = mlp.Forward(X, Y);
        float thr = sldThreshold.value;
        (_, _, _, _, _, _, _, float acc) = Metrics(P, Y, thr);

        float F = ComputeFaithfulness();

        _tracker.ReportAccuracy(Mathf.Clamp01(acc));
        _tracker.ReportFaithfulness(F);

        // Log snapshot for traceability
        EventLogger.Instance?.LogEvent(
            eventType: "MetricsSnapshot",
            key: SCENE_ID,
            fScore: F,
            extra: $"acc={acc:0.000};thr={thr:0.00}"
        );
        EvalLogger.Instance?.Metric("S4_MetricsSnapshot", new Dictionary<string, object>
        {
            { "fScore", F },
            { "accuracy", acc },
            { "threshold", thr }
        });
    }

    private void LogFaithfulnessSnapshot()
    {
        if (mlp == null || X == null || Y == null) return;

        float F = ComputeFaithfulness();
        EventLogger.Instance?.LogEvent(
            eventType: "FaithfulnessUpdated",
            key: SCENE_ID,
            fScore: F
        );
        EvalLogger.Instance?.Metric("S4_FaithfulnessUpdated", new Dictionary<string, object>
        {
            { "fScore", F }
        });
    }

    // Perturbation-based faithfulness: stability of correct predictions
    private float ComputeFaithfulness()
    {
        if (mlp == null || X == null || Y == null) return 0f;
        int n = X.GetLength(0);
        if (n == 0) return 0f;

        float minX = float.PositiveInfinity, maxX = float.NegativeInfinity;
        float minY = float.PositiveInfinity, maxY = float.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            float px = X[i, 0];
            float py = X[i, 1];
            if (px < minX) minX = px; if (px > maxX) maxX = px;
            if (py < minY) minY = py; if (py > maxY) maxY = py;
        }

        float scaleX = Mathf.Max(1e-4f, maxX - minX);
        float scaleY = Mathf.Max(1e-4f, maxY - minY);
        float sigmaX = 0.05f * scaleX;
        float sigmaY = 0.05f * scaleY;

        int maxSamples = Mathf.Min(200, n);
        int kPerturb = 3;
        float sumStability = 0f;
        int stableCount = 0;

        for (int si = 0; si < maxSamples; si++)
        {
            int i = si * n / maxSamples;

            float[,] Xi = new float[1, 2] { { X[i, 0], X[i, 1] } };
            float[,] Yi = new float[1, 1] { { Y[i, 0] } };
            var (_, Pbase) = mlp.Forward(Xi, Yi);
            float p = Pbase[0, 0];
            bool pred = p >= 0.5f;
            bool lab = Y[i, 0] >= 0.5f;
            if (pred != lab) continue;

            float deltaSum = 0f;
            for (int k = 0; k < kPerturb; k++)
            {
                float dx = NextGaussian() * sigmaX;
                float dy = NextGaussian() * sigmaY;
                float nx = Mathf.Clamp(X[i, 0] + dx, minX, maxX);
                float ny = Mathf.Clamp(X[i, 1] + dy, minY, maxY);

                float[,] Xp = new float[1, 2] { { nx, ny } };
                var (_, Pp) = mlp.Forward(Xp, Yi);
                float pp = Pp[0, 0];
                deltaSum += Mathf.Abs(pp - p);
            }

            float avgDelta = deltaSum / kPerturb;
            float stability = Mathf.Clamp01(1f - avgDelta);
            sumStability += stability;
            stableCount++;
        }

        if (stableCount == 0) return 0f;
        return sumStability / stableCount;
    }

    private float NextGaussian()
    {
        double u1 = 1.0 - rnd.NextDouble();
        double u2 = 1.0 - rnd.NextDouble();
        double randStdNormal =
            Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return (float)randStdNormal;
    }

    // ---------- Run summary & buttons ----------

    private void RecordRunSummary(string reason)
    {
        if (stepCount <= 0 || mlp == null || X == null || Y == null)
            return;

        var (_, P) = mlp.Forward(X, Y);
        float thr = sldThreshold.value;
        (_, _, _, _, _, _, _, float acc) = Metrics(P, Y, thr);
        float F = ComputeFaithfulness();
        bool success = F >= SUCCESS_F_THRESHOLD;

        CrossSceneComparisonManager.Instance?.RegisterRun(SCENE_ID, F, success);

        EvalLogger.Instance?.Metric("S4_RunSummary", new Dictionary<string, object>
        {
            { "fScore", F },
            { "accuracy", acc },
            { "success", success },
            { "reason", reason },
            { "steps", stepCount },
            { "elapsedSeconds", elapsed },
            { "threshold", thr }
        });

        EventLogger.Instance?.LogEvent(
            eventType: "RunSummaryLocal",
            key: reason,
            fScore: F,
            extra: $"scene={SCENE_ID};success={success};acc={acc:0.000};steps={stepCount};t={elapsed:0.0};thr={thr:0.00}"
        );
    }

    // Hook this to your "Next/Done" button in S4.
    public void OnSceneCompleted()
    {
        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();

        var (_, P) = mlp.Forward(X, Y);
        float thr = sldThreshold.value;
        (_, _, _, _, _, _, _, float acc) = Metrics(P, Y, thr);
        float F = ComputeFaithfulness();
        bool success = F >= SUCCESS_F_THRESHOLD;

        _tracker?.ReportSceneFinish();

        CrossSceneComparisonManager.Instance?.RegisterRun(SCENE_ID, F, success);

        EvalLogger.Instance?.Metric("S4_RunCompleted", new Dictionary<string, object>
        {
            { "fScore", F },
            { "accuracy", acc },
            { "success", success },
            { "steps", stepCount },
            { "elapsedSeconds", elapsed },
            { "threshold", thr }
        });

        EventLogger.Instance?.LogEvent(
            eventType: "RunCompleted",
            key: SCENE_ID,
            value: success ? "success" : "fail",
            fScore: F,
            extra: $"acc={acc:0.000};steps={stepCount};t={elapsed:0.0};thr={thr:0.00}"
        );
    }
}
