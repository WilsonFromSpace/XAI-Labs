using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System;
using System.Collections.Generic;
using System.Globalization;

public class OptimizersExplorer : MonoBehaviour
{
    private const string SCENE_ID = "S2_Optimizers";
    private const float SUCCESS_F_THRESHOLD = 0.85f;

    [Header("Refs")]
    public Dataset2D dataset;                 // assign asset; cloned at runtime
    public DecisionFieldRenderer field;       // shows the "focused" model
    public Transform pointsParent;
    public GameObject pointPrefab;

    [Header("UI")]
    public Button btnStep;
    public Toggle tglAuto;
    public TMP_Dropdown drpFocus;             // 0=SGD,1=Momentum,2=Adam
    public TMP_Dropdown drpActivation;        // hidden activation
    public Toggle tglMinibatch;
    public Slider sldBatch;
    public Slider sldLR;
    public Slider sldMomentum;                // β
    public Slider sldAdamB1;                  // β1
    public Slider sldAdamB2;                  // β2
    public TMP_Text txtLoss;
    public Button btnReset;
    public Button btnShuffle;

    [Header("Charts")]
    public LossChartMulti chart;

    // models & optimizers
    MLP mlpSGD, mlpMom, mlpAdam;
    IOptimizer optSGD = new OptSGD();
    OptMomentum optMom = new OptMomentum();
    OptAdam optAdam = new OptAdam();

    float[,] X, Y;
    readonly List<GameObject> pointGOs = new List<GameObject>();
    System.Random rnd = new System.Random(1234);

    bool auto = false;
    float timer = 0f;
    const float dt = 0.05f;
    int stepCount = 0;

    ObjectiveTracker _tracker;

    void Start()
    {
        // --- Scene start logging ---
        EventLogger.Instance?.LogEvent("SceneStart", key: SCENE_ID);
        EvalLogger.Instance?.Info("SceneStart_S2", new Dictionary<string, object>
        {
            { "sceneId", SCENE_ID }
        });

        // dataset runtime clone + random pose
        if (dataset == null)
        {
            dataset = ScriptableObject.CreateInstance<Dataset2D>();
            dataset.GenerateBlobs();
        }
        else
        {
            dataset = ScriptableObject.Instantiate(dataset);
            dataset.ReseedAndGenerateClean();
        }
        X = dataset.XMatrix();
        Y = dataset.YMatrix();

        // init 3 identical models from same seed/weights
        var seed = UnityEngine.Random.Range(1, 1_000_000);
        mlpSGD = new MLP(2, 3, 1, seed)
        {
            lossType = LossType.BCE,
            activation = (Act)drpActivation.value
        };
        mlpMom = CloneMLP(mlpSGD);
        mlpAdam = CloneMLP(mlpSGD);

        // optimizer params
        optMom.beta = sldMomentum.value;
        optAdam.beta1 = sldAdamB1.value;
        optAdam.beta2 = sldAdamB2.value;

        _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();

        // --- UI hooks ---

        // Manual step
        btnStep.onClick.AddListener(() =>
        {
            Step();
            _tracker?.ReportAction("step_train");
            EventLogger.Instance?.LogEvent("Action", key: "step_train");
            EvalLogger.Instance?.ActionEvent("S2_StepManual", new Dictionary<string, object>
            {
                { "step", stepCount }
            });
        });

        // Auto toggle
        tglAuto.onValueChanged.AddListener(v =>
        {
            auto = v;
            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "auto_mode",
                value: v ? "on" : "off"
            );
            EvalLogger.Instance?.ActionEvent("S2_ToggleAuto", new Dictionary<string, object>
            {
                { "isOn", v }
            });
        });

        // Focused optimizer (SGD/Mom/Adam)
        drpFocus.onValueChanged.AddListener(OnFocusChanged);

        // Hidden activation
        drpActivation.onValueChanged.AddListener(OnActivationChanged);

        // Minibatch toggle / size
        if (tglMinibatch)
        {
            tglMinibatch.onValueChanged.AddListener(v =>
            {
                EventLogger.Instance?.LogEvent(
                    eventType: "ParamChange",
                    key: "minibatch_enabled",
                    value: v ? "true" : "false"
                );
                EvalLogger.Instance?.ActionEvent("S2_ToggleMinibatch", new Dictionary<string, object>
                {
                    { "isOn", v }
                });
            });
        }

        if (sldBatch)
        {
            sldBatch.onValueChanged.AddListener(v =>
            {
                EventLogger.Instance?.LogEvent(
                    eventType: "ParamChange",
                    key: "batch_size",
                    value: ((int)v).ToString()
                );
                EvalLogger.Instance?.ActionEvent("S2_ChangeBatchSize", new Dictionary<string, object>
                {
                    { "batchSize", (int)v }
                });
            });
        }

        // LR + optimizer hyperparams
        sldLR.onValueChanged.AddListener(v =>
        {
            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "learning_rate",
                value: v.ToString("F4", CultureInfo.InvariantCulture)
            );
            EvalLogger.Instance?.ActionEvent("S2_ChangeLR", new Dictionary<string, object>
            {
                { "lr", v }
            });
        });

        sldMomentum.onValueChanged.AddListener(v =>
        {
            optMom.beta = v;
            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "momentum_beta",
                value: v.ToString("F3", CultureInfo.InvariantCulture)
            );
            EvalLogger.Instance?.ActionEvent("S2_ChangeMomentum", new Dictionary<string, object>
            {
                { "beta", v }
            });
        });

        sldAdamB1.onValueChanged.AddListener(v =>
        {
            optAdam.beta1 = v;
            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "adam_beta1",
                value: v.ToString("F3", CultureInfo.InvariantCulture)
            );
            EvalLogger.Instance?.ActionEvent("S2_ChangeAdamB1", new Dictionary<string, object>
            {
                { "beta1", v }
            });
        });

        sldAdamB2.onValueChanged.AddListener(v =>
        {
            optAdam.beta2 = v;
            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "adam_beta2",
                value: v.ToString("F3", CultureInfo.InvariantCulture)
            );
            EvalLogger.Instance?.ActionEvent("S2_ChangeAdamB2", new Dictionary<string, object>
            {
                { "beta2", v }
            });
        });

        // Reset & Shuffle
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

        // points & field & initial losses
        SpawnPoints();
        RedrawField();
        PushInitialLossesAndReport();
    }

    void Update()
    {
        if (!auto) return;

        timer += Time.deltaTime;
        if (timer >= dt)
        {
            timer = 0f;
            Step();
            _tracker?.ReportAction("step_train_auto");

            EventLogger.Instance?.LogEvent(
                eventType: "Action",
                key: "step_train_auto"
            );
            EvalLogger.Instance?.ActionEvent("S2_StepAuto", new Dictionary<string, object>
            {
                { "step", stepCount }
            });
        }
    }

    void OnDestroy()
    {
        if (stepCount > 0)
        {
            RecordRunSummary("SceneExit");
        }
    }

    // ---------------- Core interactions ----------------

    void OnFocusChanged(int idx)
    {
        RedrawField();

        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();

        string name = idx == 0 ? "SGD" : idx == 1 ? "Momentum" : "Adam";
        _tracker?.ReportTriedVariant(name);

        EventLogger.Instance?.LogEvent(
            eventType: "ParamChange",
            key: "focus_optimizer",
            value: name
        );
        EvalLogger.Instance?.ActionEvent("S2_ChangeFocus", new Dictionary<string, object>
        {
            { "focus", name }
        });
    }

    void OnActivationChanged(int idx)
    {
        var act = (Act)idx;

        mlpSGD.activation = act;
        mlpMom.activation = act;
        mlpAdam.activation = act;

        RedrawField();
        PushLossesAndReport();

        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();
        _tracker?.ReportTriedVariant(act.ToString());

        EventLogger.Instance?.LogEvent(
            eventType: "ParamChange",
            key: "activation",
            value: act.ToString()
        );
        EvalLogger.Instance?.ActionEvent("S2_ChangeActivation", new Dictionary<string, object>
        {
            { "activation", act.ToString() }
        });
    }

    void Step()
    {
        float lr = sldLR.value;

        // sample shared batch (fair comparison)
        float[,] Xb, Yb;
        int bs = dataset.count;
        if (tglMinibatch && tglMinibatch.isOn)
        {
            bs = Mathf.Clamp((int)sldBatch.value, 1, dataset.count);
            var idx = SampleBatch(bs, dataset.count);
            (Xb, Yb) = GatherBatch(idx);
        }
        else
        {
            (Xb, Yb) = (X, Y);
        }

        // forward + backprop → apply optimizer updates
        mlpSGD.Forward(Xb, Yb); optSGD.Apply(mlpSGD.Ls, lr, bs);
        mlpMom.Forward(Xb, Yb); optMom.Apply(mlpMom.Ls, lr, bs);
        mlpAdam.Forward(Xb, Yb); optAdam.Apply(mlpAdam.Ls, lr, bs);

        RedrawField();
        PushLossesAndReport();

        stepCount++;

        EventLogger.Instance?.LogEvent(
            eventType: "StepTrain",
            key: "step",
            value: stepCount.ToString()
        );
        EvalLogger.Instance?.Metric("S2_StepTrain", new Dictionary<string, object>
        {
            { "step", stepCount }
        });
    }

    void RedrawField()
    {
        field.Redraw(FocusedProb);
    }

    float FocusedProb(Vector2 w)
    {
        float[,] Xi = new float[1, 2] { { w.x, w.y } };
        float[,] Yi = new float[1, 1] { { 0f } };
        var choice = drpFocus.value; // 0=SGD,1=Momentum,2=Adam

        if (choice == 0) return mlpSGD.Forward(Xi, Yi).Item2[0, 0];
        if (choice == 1) return mlpMom.Forward(Xi, Yi).Item2[0, 0];
        return mlpAdam.Forward(Xi, Yi).Item2[0, 0];
    }

    // ---------------- Loss, chart & gamification ----------------

    void PushInitialLossesAndReport()
    {
        var lS = mlpSGD.Forward(X, Y).loss;
        var lM = mlpMom.Forward(X, Y).loss;
        var lA = mlpAdam.Forward(X, Y).loss;

        chart?.Push(lS, lM, lA);
        UpdateLossLabelFromCurrentFocus();
        ReportLossObjective(lS, lM, lA);
        ReportFaithfulnessForBestModel();
    }

    void PushLossesAndReport()
    {
        var lS = mlpSGD.Forward(X, Y).loss;
        var lM = mlpMom.Forward(X, Y).loss;
        var lA = mlpAdam.Forward(X, Y).loss;

        chart?.Push(lS, lM, lA);
        UpdateLossLabelFromCurrentFocus();
        ReportLossObjective(lS, lM, lA);
        ReportFaithfulnessForBestModel();
    }

    void ReportLossObjective(float lSGD, float lMom, float lAdam)
    {
        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();
        if (_tracker == null) return;

        float best = Mathf.Min(lSGD, Mathf.Min(lMom, lAdam));
        _tracker.ReportLoss(best);
    }

    void UpdateLossLabelFromCurrentFocus()
    {
        if (txtLoss == null) return;

        var choice = drpFocus.value;
        string name;
        float curr;

        if (choice == 0)
        {
            name = "SGD";
            curr = mlpSGD.Forward(X, Y).loss;
        }
        else if (choice == 1)
        {
            name = "Momentum";
            curr = mlpMom.Forward(X, Y).loss;
        }
        else
        {
            name = "Adam";
            curr = mlpAdam.Forward(X, Y).loss;
        }

        txtLoss.text = $"[{name}] Loss: {curr:F4} | LR: {sldLR.value:F4}";
    }

    // ----- utils -----

    void SpawnPoints()
    {
        for (int i = pointsParent.childCount - 1; i >= 0; i--)
            Destroy(pointsParent.GetChild(i).gameObject);
        pointGOs.Clear();

        for (int i = 0; i < dataset.points.Length; i++)
        {
            var go = Instantiate(pointPrefab, dataset.points[i], Quaternion.identity, pointsParent);
            var sr = go.GetComponent<SpriteRenderer>();
            sr.color = dataset.labels[i] > 0.5f
                ? new Color(0.94f, 0.28f, 0.28f)
                : new Color(0.28f, 0.46f, 0.94f);
            sr.sortingOrder = 10;
            go.transform.localScale = Vector3.one * 0.15f;
            pointGOs.Add(go);
        }
    }

    void ResetAll()
    {
        var seed = UnityEngine.Random.Range(1, 1_000_000);
        mlpSGD = new MLP(2, 3, 1, seed)
        {
            lossType = LossType.BCE,
            activation = (Act)drpActivation.value
        };
        mlpMom = CloneMLP(mlpSGD);
        mlpAdam = CloneMLP(mlpSGD);

        optMom.beta = sldMomentum.value;
        optAdam.beta1 = sldAdamB1.value;
        optAdam.beta2 = sldAdamB2.value;

        optSGD.Reset(mlpSGD.Ls);
        optMom.Reset(mlpMom.Ls);
        optAdam.Reset(mlpAdam.Ls);

        chart?.ClearSeries();
        RedrawField();
        PushInitialLossesAndReport();

        stepCount = 0;

        EventLogger.Instance?.LogEvent("ResetPressed");
        EvalLogger.Instance?.ActionEvent("S2_Reset", null);
    }

    void ShuffleData()
    {
        dataset.ReseedAndGenerateClean();
        X = dataset.XMatrix();
        Y = dataset.YMatrix();

        SpawnPoints();
        chart?.ClearSeries();
        RedrawField();
        PushInitialLossesAndReport();

        stepCount = 0;

        EventLogger.Instance?.LogEvent("ShuffleData");
        EvalLogger.Instance?.ActionEvent("S2_ShuffleData", null);
    }

    int[] SampleBatch(int bs, int total)
    {
        var set = new HashSet<int>();
        while (set.Count < bs) set.Add(rnd.Next(total));
        var arr = new int[bs];
        int k = 0;
        foreach (var i in set) arr[k++] = i;
        return arr;
    }

    (float[,], float[,]) GatherBatch(int[] idx)
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

    MLP CloneMLP(MLP src)
    {
        var dst = new MLP(
            src.Ls[0].W.GetLength(0),
            src.Ls[0].W.GetLength(1),
            src.Ls[1].W.GetLength(1),
            seed: 1
        );
        dst.lossType = src.lossType;
        dst.activation = src.activation;
        dst.lr = src.lr;

        dst.Ls[0].W = Copy2D(src.Ls[0].W);
        dst.Ls[0].b = Copy1D(src.Ls[0].b);
        dst.Ls[1].W = Copy2D(src.Ls[1].W);
        dst.Ls[1].b = Copy1D(src.Ls[1].b);

        return dst;
    }

    static float[,] Copy2D(float[,] A)
    {
        int n = A.GetLength(0), m = A.GetLength(1);
        var B = new float[n, m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                B[i, j] = A[i, j];
        return B;
    }

    static float[] Copy1D(float[] a)
    {
        var b = new float[a.Length];
        for (int i = 0; i < a.Length; i++) b[i] = a[i];
        return b;
    }

    // ---------- Faithfulness & Cross-scene summary ----------

    float ComputeFaithfulness(MLP model)
    {
        if (model == null || X == null || Y == null) return 0f;
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
            var (_, Pbase) = model.Forward(Xi, Yi);
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
                var (_, Pp) = model.Forward(Xp, Yi);
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

    float NextGaussian()
    {
        double u1 = 1.0 - rnd.NextDouble();
        double u2 = 1.0 - rnd.NextDouble();
        double randStdNormal =
            Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return (float)randStdNormal;
    }

    void ReportFaithfulnessForBestModel()
    {
        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();
        if (_tracker == null) return;

        float fS = ComputeFaithfulness(mlpSGD);
        float fM = ComputeFaithfulness(mlpMom);
        float fA = ComputeFaithfulness(mlpAdam);
        float bestF = Mathf.Max(fS, Mathf.Max(fM, fA));

        _tracker.ReportFaithfulness(bestF);

        EventLogger.Instance?.LogEvent(
            eventType: "FaithfulnessUpdated",
            key: SCENE_ID,
            fScore: bestF
        );
        EvalLogger.Instance?.Metric("S2_FaithfulnessUpdated", new Dictionary<string, object>
        {
            { "fScore", bestF }
        });
    }

    void RecordRunSummary(string reason)
    {
        if (stepCount <= 0)
            return; // nothing meaningful yet

        float fS = ComputeFaithfulness(mlpSGD);
        float fM = ComputeFaithfulness(mlpMom);
        float fA = ComputeFaithfulness(mlpAdam);
        float bestF = Mathf.Max(fS, Mathf.Max(fM, fA));
        bool success = bestF >= SUCCESS_F_THRESHOLD;

        CrossSceneComparisonManager.Instance?.RegisterRun(SCENE_ID, bestF, success);

        EvalLogger.Instance?.Metric("S2_RunSummary", new Dictionary<string, object>
        {
            { "fScore", bestF },
            { "success", success },
            { "reason", reason },
            { "steps", stepCount }
        });

        EventLogger.Instance?.LogEvent(
            eventType: "RunSummaryLocal",
            key: reason,
            fScore: bestF,
            extra: $"scene={SCENE_ID};success={success};steps={stepCount}"
        );
    }

    // Wire this from your "Next" / "Continue" button in S2.
    public void OnSceneCompleted()
    {
        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();
        _tracker?.ReportSceneFinish();

        RecordRunSummary("ManualComplete");
    }
}
