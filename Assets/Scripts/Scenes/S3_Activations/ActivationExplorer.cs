using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using TMPro;
using System;
using System.Collections.Generic;
using System.Globalization;

public class ActivationExplorer : MonoBehaviour
{
    private const string SCENE_ID = "S3_Activation";
    private const float SUCCESS_F_THRESHOLD = 0.85f;

    [Header("Data & World")]
    public Dataset2D dataset;              // assign asset; we clone it at runtime
    public Transform pointsParent;         // World/Points
    public GameObject pointPrefab;         // your dot prefab

    [Header("Core UI")]
    public Button btnStep;
    public Toggle tglAuto;
    public Toggle tglMinibatch;
    public Slider sldBatch;
    public Slider sldLR;
    public TMP_Dropdown drpActivation;     // Tanh / ReLU / Sigmoid
    public TMP_Text txtLoss;
    public Button btnReset;
    public Button btnShuffle;
    public Toggle tglShowBoundary;         // show/hide 0.5 decision curve

    [Header("Overlays / Panels (optional)")]
    public ActivationHingesOverlay hinges;         // per-neuron hinge lines + bands
    public DecisionBoundaryCurve boundaryCurve;    // crisp GL 0.5 isocontour
    public DecisionFieldRenderer boundarySampler;  // fallback sampler if you prefer
    public ActivationCurvePanel actCurve;          // φ and φ′ mini-plot (RawImage)
    public ActivationStatsPanel actStats;          // saturation %, dead ReLU, mean |∂L/∂z|
    public ActivationExplainerPanel explainer;     // beginner narration panel
    public Toggle tglBeginner;                     // toggle to show/hide explainer

    [Header("Point Style")]
    [Range(0.02f, 0.30f)] public float dotScale = 0.10f;  // smaller = cleaner
    public bool fadeByConfidence = true;
    [Range(0f, 1f)] public float alphaMin = 0.25f;        // alpha at p≈0.5
    [Range(0f, 1f)] public float alphaMax = 0.85f;        // alpha at confident (p≈0 or 1)
    public bool desaturateUncertain = true;
    [Range(0f, 1f)] public float desaturateStrength = 0.50f;
    public Color blueColor = new Color(0.45f, 0.62f, 0.94f, 1f); // calmer blue
    public Color redColor = new Color(0.94f, 0.45f, 0.45f, 1f);  // calmer red

    // --- internals ---
    private MLP mlp;
    private float[,] X, Y;
    private readonly List<GameObject> pointGOs = new List<GameObject>();
    private System.Random rnd = new System.Random(42);

    private bool auto = false;
    private float timer = 0f;
    private const float autoDt = 0.05f;

    // --- Gamification / logging ---
    private ObjectiveTracker _tracker;
    private float elapsed = 0f;
    private int stepCount = 0;

    private void Start()
    {
        // --- Scene start logging ---
        EventLogger.Instance?.LogEvent("SceneStart", key: SCENE_ID);
        EvalLogger.Instance?.Info("SceneStart_S3", new Dictionary<string, object>
        {
            { "sceneId", SCENE_ID }
        });

        // Dataset: runtime clone + randomized pose for each run
        dataset = dataset
            ? ScriptableObject.Instantiate(dataset)
            : ScriptableObject.CreateInstance<Dataset2D>();

        if (dataset.points == null || dataset.points.Length == 0)
            dataset.GenerateBlobs();

        dataset.ReseedAndGenerateClean();
        X = dataset.XMatrix();
        Y = dataset.YMatrix();

        // Model: 2→3→1, BCE, hidden activation from dropdown
        mlp = new MLP(2, 3, 1, seed: UnityEngine.Random.Range(1, 1_000_000))
        {
            lossType = LossType.BCE,
            activation = (Act)drpActivation.value,
            lr = sldLR ? sldLR.value : 0.1f
        };

        // Gamification tracker
        _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();

        // --- UI bindings ---

        if (btnStep)
            btnStep.onClick.AddListener(() =>
            {
                Step();
                _tracker?.ReportAction("step_train");
                EventLogger.Instance?.LogEvent("Action", key: "step_train");
                EvalLogger.Instance?.ActionEvent("S3_StepManual", new Dictionary<string, object>
                {
                    { "step", stepCount }
                });
            });

        if (tglAuto)
            tglAuto.onValueChanged.AddListener(v =>
            {
                auto = v;
                EventLogger.Instance?.LogEvent(
                    eventType: "ParamChange",
                    key: "auto_mode",
                    value: v ? "on" : "off"
                );
                EvalLogger.Instance?.ActionEvent("S3_ToggleAuto", new Dictionary<string, object>
                {
                    { "isOn", v }
                });
            });

        if (tglMinibatch)
            tglMinibatch.onValueChanged.AddListener(v =>
            {
                EventLogger.Instance?.LogEvent(
                    eventType: "ParamChange",
                    key: "minibatch_enabled",
                    value: v ? "true" : "false"
                );
                EvalLogger.Instance?.ActionEvent("S3_ToggleMinibatch", new Dictionary<string, object>
                {
                    { "isOn", v }
                });
            });

        if (sldBatch)
            sldBatch.onValueChanged.AddListener(v =>
            {
                EventLogger.Instance?.LogEvent(
                    eventType: "ParamChange",
                    key: "batch_size",
                    value: ((int)v).ToString()
                );
                EvalLogger.Instance?.ActionEvent("S3_ChangeBatchSize", new Dictionary<string, object>
                {
                    { "batchSize", (int)v }
                });
            });

        if (sldLR)
            sldLR.onValueChanged.AddListener(v =>
            {
                mlp.lr = v;
                EventLogger.Instance?.LogEvent(
                    eventType: "ParamChange",
                    key: "learning_rate",
                    value: v.ToString("F4", CultureInfo.InvariantCulture)
                );
                EvalLogger.Instance?.ActionEvent("S3_ChangeLR", new Dictionary<string, object>
                {
                    { "lr", v }
                });
            });

        if (drpActivation)
            drpActivation.onValueChanged.AddListener(OnActivationChanged);

        if (btnReset)
            btnReset.onClick.AddListener(() =>
            {
                RecordRunSummary("Reset");
                ResetModel();
            });

        if (btnShuffle)
            btnShuffle.onClick.AddListener(() =>
            {
                RecordRunSummary("Shuffle");
                ShuffleData();
            });

        if (tglShowBoundary)
            tglShowBoundary.onValueChanged.AddListener(_ => Redraw());

        if (tglBeginner)
            tglBeginner.onValueChanged.AddListener(v =>
            {
                if (explainer) explainer.gameObject.SetActive(v);
                EvalLogger.Instance?.ActionEvent("S3_ToggleBeginner", new Dictionary<string, object>
                {
                    { "isOn", v }
                });
            });

        // Points
        SpawnPoints();

        // Prime mini-plot & explainer
        actCurve?.SetAct((Act)drpActivation.value);
        if (explainer)
            explainer.gameObject.SetActive(tglBeginner ? tglBeginner.isOn : true);

        // Initial draw & styles
        Redraw();
        UpdateLossText();
        UpdatePointStyles();

        // Initial metrics
        ReportMetricsToTracker();
    }

    private void Update()
    {
        elapsed += Time.deltaTime;

        if (auto)
        {
            timer += Time.deltaTime;
            if (timer >= autoDt)
            {
                timer = 0f;
                Step();
                _tracker?.ReportAction("step_train_auto");
                EventLogger.Instance?.LogEvent("Action", key: "step_train_auto");
                EvalLogger.Instance?.ActionEvent("S3_StepAuto", new Dictionary<string, object>
                {
                    { "step", stepCount }
                });
            }
        }

        // Shortcuts
        if (Input.GetKeyDown(KeyCode.Space)) Step();
        if (Input.GetKeyDown(KeyCode.A) && tglAuto) tglAuto.isOn = !tglAuto.isOn;
        if (Input.GetKeyDown(KeyCode.R))
        {
            RecordRunSummary("Reset_Key");
            ResetModel();
        }
        if (Input.GetKeyDown(KeyCode.S))
        {
            RecordRunSummary("Shuffle_Key");
            ShuffleData();
        }
        if (Input.GetKeyDown(KeyCode.B) && tglShowBoundary) tglShowBoundary.isOn = !tglShowBoundary.isOn;
        if (Input.GetKeyDown(KeyCode.H) && tglBeginner) tglBeginner.isOn = !tglBeginner.isOn;
    }

    private void OnDestroy()
    {
        if (stepCount > 0)
        {
            RecordRunSummary("SceneExit");
        }
    }

    // ---------------- Core logic ----------------

    private void OnActivationChanged(int idx)
    {
        mlp.activation = (Act)idx;
        actCurve?.SetAct((Act)idx);
        Redraw();
        UpdateLossText();
        UpdatePointStyles();

        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();

        _tracker?.ReportTriedVariant(((Act)idx).ToString());

        EventLogger.Instance?.LogEvent(
            eventType: "ParamChange",
            key: "activation",
            value: ((Act)idx).ToString()
        );
        EvalLogger.Instance?.ActionEvent("S3_ChangeActivation", new Dictionary<string, object>
        {
            { "activation", ((Act)idx).ToString() }
        });

        ReportMetricsToTracker();
    }

    private void Step()
    {
        int N = dataset.count;
        float loss;

        if (tglMinibatch && tglMinibatch.isOn)
        {
            int bs = Mathf.Clamp((int)sldBatch.value, 1, N);
            var idx = SampleBatch(bs, N);
            var (Xb, Yb) = GatherBatch(idx);
            (loss, _) = mlp.Forward(Xb, Yb);
            mlp.StepSGD(bs);
        }
        else
        {
            (loss, _) = mlp.Forward(X, Y);
            mlp.StepSGD(N);
        }

        stepCount++;

        if (txtLoss)
            txtLoss.text = $"Loss: {loss:F4} | LR: {mlp.lr:F4} | Act: {mlp.activation}";

        Redraw();
        UpdatePointStyles();

        EventLogger.Instance?.LogEvent(
            eventType: "StepTrain",
            key: "step",
            value: stepCount.ToString()
        );
        EvalLogger.Instance?.Metric("S3_StepTrain", new Dictionary<string, object>
        {
            { "step", stepCount },
            { "loss", loss }
        });

        ReportMetricsToTracker();
    }

    private void Redraw()
    {
        // Per-neuron geometry
        hinges?.Render(mlp, dataset);

        // 0.5 decision curve
        bool showBoundary = tglShowBoundary ? tglShowBoundary.isOn : true;

        if (showBoundary && boundaryCurve)
            boundaryCurve.Redraw(WorldToProb);

        if (boundarySampler)
        {
            boundarySampler.enabled = showBoundary && !boundaryCurve;
            if (boundarySampler.enabled)
                boundarySampler.Redraw(WorldToProb);
        }

        // Stats + explainer
        actStats?.UpdateFrom(mlp, dataset);
        explainer?.UpdateFrom(mlp, dataset);
    }

    private float WorldToProb(Vector2 w)
    {
        float[,] Xi = new float[1, 2] { { w.x, w.y } };
        float[,] Yi = new float[1, 1] { { 0f } };
        var (_, pred) = mlp.Forward(Xi, Yi);
        return Mathf.Clamp01(pred[0, 0]);
    }

    private void UpdateLossText()
    {
        var (loss, _) = mlp.Forward(X, Y);
        if (txtLoss)
            txtLoss.text = $"Loss: {loss:F4} | LR: {mlp.lr:F4} | Act: {mlp.activation}";
    }

    private void ResetModel()
    {
        mlp = new MLP(2, 3, 1, seed: UnityEngine.Random.Range(1, 1_000_000))
        {
            lossType = LossType.BCE,
            activation = (Act)drpActivation.value,
            lr = sldLR ? sldLR.value : mlp.lr
        };

        actCurve?.SetAct((Act)drpActivation.value);

        stepCount = 0;
        elapsed = 0f;

        Redraw();
        UpdateLossText();
        UpdatePointStyles();

        ReportMetricsToTracker();

        EventLogger.Instance?.LogEvent("ResetPressed");
        EvalLogger.Instance?.ActionEvent("S3_Reset", null);
    }

    private void ShuffleData()
    {
        dataset.ReseedAndGenerateClean();
        X = dataset.XMatrix();
        Y = dataset.YMatrix();

        SpawnPoints();

        stepCount = 0;
        elapsed = 0f;

        Redraw();
        UpdateLossText();
        UpdatePointStyles();

        ReportMetricsToTracker();

        EventLogger.Instance?.LogEvent("ShuffleData");
        EvalLogger.Instance?.ActionEvent("S3_ShuffleData", null);
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

    private void UpdatePointStyles()
    {
        if (pointGOs.Count == 0) return;

        float[,] P = null;
        if (mlp != null && (fadeByConfidence || desaturateUncertain))
            P = mlp.Forward(X, Y).Item2;

        for (int i = 0; i < pointGOs.Count; i++)
        {
            var sr = pointGOs[i].GetComponent<SpriteRenderer>();
            bool isRed = dataset.labels[i] > 0.5f;
            Color baseCol = isRed ? redColor : blueColor;

            float conf = 1f;
            if (P != null)
                conf = Mathf.Clamp01(Mathf.Abs(P[i, 0] - 0.5f) * 2f);

            float a = fadeByConfidence ? Mathf.Lerp(alphaMin, alphaMax, conf) : alphaMax;

            if (desaturateUncertain)
            {
                float g = baseCol.grayscale;
                Color gray = new Color(g, g, g, 1f);
                float t = desaturateStrength * (1f - conf);
                baseCol = Color.Lerp(baseCol, gray, t);
            }

            baseCol.a = a;
            sr.color = baseCol;

            pointGOs[i].transform.localScale = Vector3.one * dotScale;
        }
    }

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

    // ---------- Metrics: Accuracy + Faithfulness ----------

    private float ComputeAccuracy()
    {
        if (mlp == null || X == null || Y == null) return 0f;
        var (_, P) = mlp.Forward(X, Y);
        int n = P.GetLength(0), correct = 0;
        for (int i = 0; i < n; i++)
        {
            bool pred = P[i, 0] >= 0.5f;
            bool lab = Y[i, 0] >= 0.5f;
            if (pred == lab) correct++;
        }
        return n > 0 ? (float)correct / n : 0f;
    }

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

    private void ReportMetricsToTracker()
    {
        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();
        if (_tracker == null) return;

        float acc = ComputeAccuracy();
        float F = ComputeFaithfulness();

        _tracker.ReportAccuracy(acc);
        _tracker.ReportFaithfulness(F);

        EventLogger.Instance?.LogEvent(
            eventType: "FaithfulnessUpdated",
            key: SCENE_ID,
            fScore: F
        );
        EvalLogger.Instance?.Metric("S3_FaithfulnessUpdated", new Dictionary<string, object>
        {
            { "fScore", F },
            { "accuracy", acc }
        });
    }

    // ---------- Run summary & external hook ----------

    private void RecordRunSummary(string reason)
    {
        if (stepCount <= 0 || mlp == null || X == null || Y == null)
            return;

        float acc = ComputeAccuracy();
        float F = ComputeFaithfulness();
        bool success = F >= SUCCESS_F_THRESHOLD;

        CrossSceneComparisonManager.Instance?.RegisterRun(SCENE_ID, F, success);

        EvalLogger.Instance?.Metric("S3_RunSummary", new Dictionary<string, object>
        {
            { "fScore", F },
            { "accuracy", acc },
            { "success", success },
            { "reason", reason },
            { "steps", stepCount },
            { "elapsedSeconds", elapsed }
        });

        EventLogger.Instance?.LogEvent(
            eventType: "RunSummaryLocal",
            key: reason,
            fScore: F,
            extra: $"scene={SCENE_ID};success={success};acc={acc:0.000};steps={stepCount};t={elapsed:0.0}"
        );
    }

    // Wire this from your "Next"/"Continue" button in S3.
    public void OnSceneCompleted()
    {
        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();

        float acc = ComputeAccuracy();
        float F = ComputeFaithfulness();
        bool success = F >= SUCCESS_F_THRESHOLD;

        _tracker?.ReportSceneFinish();

        CrossSceneComparisonManager.Instance?.RegisterRun(SCENE_ID, F, success);

        EvalLogger.Instance?.Metric("S3_RunCompleted", new Dictionary<string, object>
        {
            { "fScore", F },
            { "accuracy", acc },
            { "success", success },
            { "steps", stepCount },
            { "elapsedSeconds", elapsed }
        });

        EventLogger.Instance?.LogEvent(
            eventType: "RunCompleted",
            key: SCENE_ID,
            value: success ? "success" : "fail",
            fScore: F,
            extra: $"acc={acc:0.000};steps={stepCount};t={elapsed:0.0}"
        );
    }
}
