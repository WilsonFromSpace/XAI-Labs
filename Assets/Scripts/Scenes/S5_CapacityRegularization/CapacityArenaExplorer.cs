using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System;
using System.Collections.Generic;
using System.Globalization;

public class CapacityArenaExplorer : MonoBehaviour
{
    [Header("Data & World")]
    public Dataset2D dataset;
    [Range(0.05f, 0.5f)] public float valSplit = 0.25f;
    public Transform pointsParent;
    public GameObject pointPrefab;

    [Header("Capacity")]
    public Slider sldUnits;
    public TMP_Dropdown drpLayers;
    public Button btnRebuild;

    [Header("Regularization")]
    public Slider sldL1, sldL2, sldDrop;
    public TMP_Text txtSparsity;

    [Header("Training UI")]
    public Button btnStep;
    public Toggle tglAuto, tglMinibatch;
    public Slider sldBatch, sldLR;
    public Button btnReset, btnShuffle;
    public TMP_Text txtTrain, txtVal, txtScore;

    [Header("Panels")]
    public WeightSkylinePanel skyline;
    public DropoutRainPanel dropoutRain;
    public GapGaugePanel gapGauge;

    [Header("Overlays (new)")]
    public DecisionContourPanel decision;              // world overlay
    public bool highlightMistakes = true;
    public Color mistakeColor = new Color(1f, 0.85f, 0.2f, 1f);
    [Range(0.05f, 0.2f)] public float normalDotScale = 0.09f;
    [Range(0.06f, 0.25f)] public float mistakeDotScale = 0.13f;

    // internals
    MLP_Capacity mlp;
    float[,] Xtr, Ytr, Xval, Yval;
    readonly List<SpriteRenderer> dotSR = new();
    System.Random rnd = new System.Random(123);
    bool auto = false;
    float t = 0f;
    const float dt = 0.05f;
    Vector2 wmin, wmax;
    int stepCount = 0;

    // Gamification
    ObjectiveTracker _tracker;
    readonly HashSet<string> _explored = new HashSet<string>();

    void Start()
    {
        // Log scene start
        EventLogger.Instance?.LogEvent(
            eventType: "SceneStart",
            key: UnityEngine.SceneManagement.SceneManager.GetActiveScene().name
        );

        InitData();
        BuildModel();
        WireUI();
        SpawnDots();
        ConfigureBounds();
        RedrawAll();

        _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();

        // Initial metrics -> tracker + log
        ReportMetricsToTrackerFromCurrent();
        LogFaithfulnessSnapshot();
    }

    void InitData()
    {
        dataset = dataset
            ? ScriptableObject.Instantiate(dataset)
            : ScriptableObject.CreateInstance<Dataset2D>();

        if (dataset.points == null || dataset.points.Length == 0)
            dataset.GenerateBlobs();

        dataset.ReseedAndGenerateClean();

        var X = dataset.XMatrix();
        var Y = dataset.YMatrix();
        int N = dataset.count;
        int Nval = Mathf.RoundToInt(N * valSplit);
        int Ntr = N - Nval;

        int[] idx = new int[N];
        for (int i = 0; i < N; i++) idx[i] = i;
        for (int i = 0; i < N; i++)
        {
            int j = rnd.Next(N);
            (idx[i], idx[j]) = (idx[j], idx[i]);
        }

        Xtr = new float[Ntr, 2];
        Ytr = new float[Ntr, 1];
        Xval = new float[Nval, 2];
        Yval = new float[Nval, 1];

        for (int k = 0; k < Ntr; k++)
        {
            int i = idx[k];
            Xtr[k, 0] = X[i, 0];
            Xtr[k, 1] = X[i, 1];
            Ytr[k, 0] = Y[i, 0];
        }
        for (int k = 0; k < Nval; k++)
        {
            int i = idx[Ntr + k];
            Xval[k, 0] = X[i, 0];
            Xval[k, 1] = X[i, 1];
            Yval[k, 0] = Y[i, 0];
        }
    }

    void ConfigureBounds()
    {
        float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
        foreach (var p in dataset.points)
        {
            if (p.x < minx) minx = p.x;
            if (p.y < miny) miny = p.y;
            if (p.x > maxx) maxx = p.x;
            if (p.y > maxy) maxy = p.y;
        }
        Vector2 size = new Vector2(maxx - minx, maxy - miny);
        Vector2 pad = 0.15f * size;
        wmin = new Vector2(minx, miny) - pad;
        wmax = new Vector2(maxx, maxy) + pad;
        if (decision) decision.Configure(wmin, wmax);
    }

    void BuildModel()
    {
        int H = Mathf.Max(2, (int)sldUnits.value);
        int L = Mathf.Clamp(drpLayers.value + 1, 1, 3);

        mlp = new MLP_Capacity(2, H, 1, layers: L, seed: UnityEngine.Random.Range(1, 1_000_000))
        {
            lossType = MLP_Capacity.LossType.BCE,
            activation = MLP_Capacity.Act.ReLU,
            lr = sldLR.value,
            l1 = sldL1.value,
            l2 = sldL2.value,
            dropoutP = sldDrop.value
        };
    }

    void WireUI()
    {
        // Step button
        btnStep.onClick.AddListener(() =>
        {
            Step();
            _tracker?.ReportAction("step_train");
            EventLogger.Instance?.LogEvent("Action", key: "step_train");
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
        });

        // Minibatch toggle
        tglMinibatch.onValueChanged.AddListener(v =>
        {
            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "minibatch_enabled",
                value: v ? "true" : "false"
            );
        });

        // Batch size
        sldBatch.onValueChanged.AddListener(v =>
        {
            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "batch_size",
                value: ((int)v).ToString()
            );
        });

        // Learning rate
        sldLR.onValueChanged.AddListener(v =>
        {
            mlp.lr = v;
            RedrawAll();

            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "learning_rate",
                value: v.ToString("F4", CultureInfo.InvariantCulture)
            );
        });

        // Rebuild architecture
        btnRebuild.onClick.AddListener(() =>
        {
            BuildModel();
            RedrawAll();
            RegisterExploration("Architecture");
            ReportMetricsToTrackerFromCurrent();
            LogFaithfulnessSnapshot();

            EventLogger.Instance?.LogEvent("Action", key: "rebuild_model");
        });

        // Regularization sliders
        sldL1.onValueChanged.AddListener(_ =>
        {
            mlp.l1 = sldL1.value;
            RedrawAll();
            RegisterExploration("L1");
            ReportMetricsToTrackerFromCurrent();
            LogFaithfulnessSnapshot();

            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "l1",
                value: sldL1.value.ToString("F4", CultureInfo.InvariantCulture)
            );
        });

        sldL2.onValueChanged.AddListener(_ =>
        {
            mlp.l2 = sldL2.value;
            RedrawAll();
            RegisterExploration("L2");
            ReportMetricsToTrackerFromCurrent();
            LogFaithfulnessSnapshot();

            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "l2",
                value: sldL2.value.ToString("F4", CultureInfo.InvariantCulture)
            );
        });

        sldDrop.onValueChanged.AddListener(_ =>
        {
            mlp.dropoutP = sldDrop.value;
            RedrawAll();
            RegisterExploration("Dropout");
            ReportMetricsToTrackerFromCurrent();
            LogFaithfulnessSnapshot();

            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "dropout_p",
                value: sldDrop.value.ToString("F3", CultureInfo.InvariantCulture)
            );
        });

        // Capacity sliders
        sldUnits.onValueChanged.AddListener(_ =>
        {
            RegisterExploration("Width");
            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "units",
                value: ((int)sldUnits.value).ToString()
            );
        });

        drpLayers.onValueChanged.AddListener(_ =>
        {
            RegisterExploration("Depth");
            EventLogger.Instance?.LogEvent(
                eventType: "ParamChange",
                key: "layers",
                value: (drpLayers.value + 1).ToString()
            );
        });

        // Reset
        btnReset.onClick.AddListener(() =>
        {
            BuildModel();
            RedrawAll();
            ReportMetricsToTrackerFromCurrent();
            LogFaithfulnessSnapshot();

            stepCount = 0;

            EventLogger.Instance?.LogEvent("ResetPressed");
        });

        // Shuffle
        btnShuffle.onClick.AddListener(() =>
        {
            InitData();
            ConfigureBounds();
            RedrawAll();
            ReportMetricsToTrackerFromCurrent();
            LogFaithfulnessSnapshot();

            stepCount = 0;

            EventLogger.Instance?.LogEvent("ShuffleData");
        });
    }

    void Update()
    {
        if (!auto) return;

        t += Time.deltaTime;
        if (t >= dt)
        {
            t = 0f;
            Step();
            _tracker?.ReportAction("step_train_auto");
            EventLogger.Instance?.LogEvent("Action", key: "step_train_auto");
        }
    }

    void Step()
    {
        int N = Xtr.GetLength(0);

        if (tglMinibatch && tglMinibatch.isOn)
        {
            int bs = Mathf.Clamp((int)sldBatch.value, 1, N);
            var idx = SampleBatch(bs, N);
            var (Xb, Yb) = GatherBatch(idx, Xtr, Ytr);
            mlp.Forward(Xb, Yb, train: true);
            mlp.StepSGD(bs);
        }
        else
        {
            mlp.Forward(Xtr, Ytr, train: true);
            mlp.StepSGD(N);
        }

        stepCount++;

        RedrawAll();
        ReportMetricsToTrackerFromCurrent();
        LogFaithfulnessSnapshot();

        EventLogger.Instance?.LogEvent(
            eventType: "StepTrain",
            key: "step",
            value: stepCount.ToString()
        );
    }

    void RedrawAll()
    {
        var (ltr, Ptr) = mlp.Forward(Xtr, Ytr, train: false);
        var (lva, Pva) = mlp.Forward(Xval, Yval, train: false);

        float atr = Accuracy(Ptr, Ytr, 0.5f);
        float ava = Accuracy(Pva, Yval, 0.5f);
        float gap = Mathf.Max(0f, atr - ava);
        float score = ava - 0.5f * gap; // generalization score

        if (txtTrain) txtTrain.text = $"Train: acc {atr:0.000} | loss {ltr:0.0000}";
        if (txtVal) txtVal.text = $"Val:   acc {ava:0.000} | loss {lva:0.0000}";
        if (txtScore) txtScore.text = $"Generalization Score: {score:0.000}";

        skyline?.Redraw(mlp);
        dropoutRain?.Redraw(mlp);
        gapGauge?.Redraw(gap);

        float sp = PercentZeroWeights(mlp);
        if (txtSparsity) txtSparsity.text = $"Sparsity: {sp:0.0}%";

        decision?.Redraw(mlp);
        if (highlightMistakes) UpdateDotErrors();

        // Gamification + F handled in ReportMetricsToTrackerFromCurrent()
    }

    void UpdateDotErrors()
    {
        var Xall = dataset.XMatrix();
        var Yall = dataset.YMatrix();
        var P = mlp.Forward(Xall, null, train: false).pred;

        for (int i = 0; i < dotSR.Count; i++)
        {
            bool isPos = Yall[i, 0] > 0.5f;
            bool predPos = P[i, 0] >= 0.5f;
            bool ok = (isPos == predPos);

            var baseCol = isPos
                ? new Color(0.94f, 0.45f, 0.45f, 0.75f)
                : new Color(0.45f, 0.62f, 0.94f, 0.75f);

            if (!ok)
            {
                dotSR[i].color = mistakeColor;
                dotSR[i].transform.localScale = Vector3.one * mistakeDotScale;
            }
            else
            {
                dotSR[i].color = baseCol;
                dotSR[i].transform.localScale = Vector3.one * normalDotScale;
            }
        }
    }

    // helpers

    float Accuracy(float[,] P, float[,] Y, float thr)
    {
        int N = P.GetLength(0), ok = 0;
        for (int i = 0; i < N; i++)
        {
            int h = P[i, 0] >= thr ? 1 : 0;
            int y = Y[i, 0] > 0.5f ? 1 : 0;
            if (h == y) ok++;
        }
        return ok / (float)Mathf.Max(1, N);
    }

    float PercentZeroWeights(MLP_Capacity m)
    {
        int z = 0, n = 0;
        foreach (var L in m.Ls)
        {
            for (int i = 0; i < L.W.GetLength(0); i++)
                for (int j = 0; j < L.W.GetLength(1); j++)
                {
                    n++;
                    if (Mathf.Approximately(L.W[i, j], 0f)) z++;
                }
        }
        return n > 0 ? (100f * z) / n : 0f;
    }

    void SpawnDots()
    {
        for (int i = pointsParent.childCount - 1; i >= 0; i--)
            Destroy(pointsParent.GetChild(i).gameObject);
        dotSR.Clear();

        for (int i = 0; i < dataset.points.Length; i++)
        {
            var go = Instantiate(pointPrefab, dataset.points[i], Quaternion.identity, pointsParent);
            var sr = go.GetComponent<SpriteRenderer>();
            sr.sortingOrder = 10;
            sr.color = dataset.labels[i] > 0.5f
                ? new Color(0.94f, 0.45f, 0.45f, 0.75f)
                : new Color(0.45f, 0.62f, 0.94f, 0.75f);
            go.transform.localScale = Vector3.one * normalDotScale;
            dotSR.Add(sr);
        }
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

    (float[,], float[,]) GatherBatch(int[] idx, float[,] X, float[,] Y)
    {
        var Xb = new float[idx.Length, 2];
        var Yb = new float[idx.Length, 1];
        for (int r = 0; r < idx.Length; r++)
        {
            int i = idx[r];
            Xb[r, 0] = X[i, 0];
            Xb[r, 1] = X[i, 1];
            Yb[r, 0] = Y[i, 0];
        }
        return (Xb, Yb);
    }

    // ---------- Gamification helpers ----------

    void RegisterExploration(string key)
    {
        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();
        if (_tracker == null) return;

        if (_explored.Add(key))
            _tracker.ReportTriedVariant(key);
    }

    void ReportMetricsToTrackerFromCurrent()
    {
        if (mlp == null || Xtr == null || Ytr == null || Xval == null || Yval == null)
            return;

        var (ltr, Ptr) = mlp.Forward(Xtr, Ytr, train: false);
        var (lva, Pva) = mlp.Forward(Xval, Yval, train: false);

        float atr = Accuracy(Ptr, Ytr, 0.5f);
        float ava = Accuracy(Pva, Yval, 0.5f);
        float gap = Mathf.Max(0f, atr - ava);

        float score = ava - 0.5f * gap; // can be <0, clamp later for F

        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();
        if (_tracker != null)
        {
            float F = Mathf.Clamp01(score); // interpret as F-like [0,1]
            _tracker.ReportFaithfulness(F);
        }
    }

    void LogFaithfulnessSnapshot()
    {
        if (mlp == null || Xtr == null || Ytr == null || Xval == null || Yval == null)
            return;

        var (ltr, Ptr) = mlp.Forward(Xtr, Ytr, train: false);
        var (lva, Pva) = mlp.Forward(Xval, Yval, train: false);

        float atr = Accuracy(Ptr, Ytr, 0.5f);
        float ava = Accuracy(Pva, Yval, 0.5f);
        float gap = Mathf.Max(0f, atr - ava);
        float score = ava - 0.5f * gap;

        float F = Mathf.Clamp01(score);

        EventLogger.Instance?.LogEvent(
            eventType: "FaithfulnessUpdated",
            key: UnityEngine.SceneManagement.SceneManager.GetActiveScene().name,
            fScore: F
        );
    }

    // ---------- Public hook for S5 completion ----------

    public void OnSceneCompleted()
    {
        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();

        _tracker?.ReportSceneFinish();

        string sceneId = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;

        // Final F based on latest generalization snapshot
        var (ltr, Ptr) = mlp.Forward(Xtr, Ytr, train: false);
        var (lva, Pva) = mlp.Forward(Xval, Yval, train: false);
        float atr = Accuracy(Ptr, Ytr, 0.5f);
        float ava = Accuracy(Pva, Yval, 0.5f);
        float gap = Mathf.Max(0f, atr - ava);
        float score = ava - 0.5f * gap;
        float finalF = Mathf.Clamp01(score);

        EventLogger.Instance?.LogEvent(
            eventType: "RunCompleted",
            key: sceneId,
            value: "success",
            fScore: finalF
        );

        CrossSceneComparisonManager.Instance?.RegisterRun(
            sceneId: sceneId,
            fScore: finalF,
            success: true
        );
    }
}
