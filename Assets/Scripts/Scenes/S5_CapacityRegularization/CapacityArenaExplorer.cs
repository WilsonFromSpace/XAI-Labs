using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System;
using System.Collections.Generic;

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
    public Button btnStep; public Toggle tglAuto, tglMinibatch;
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
    bool auto = false; float t = 0f; const float dt = 0.05f;
    Vector2 wmin, wmax;

    void Start()
    {
        InitData();
        BuildModel();
        WireUI();
        SpawnDots();
        ConfigureBounds();
        RedrawAll();
    }

    void InitData()
    {
        dataset = dataset ? ScriptableObject.Instantiate(dataset)
                          : ScriptableObject.CreateInstance<Dataset2D>();
        if (dataset.points == null || dataset.points.Length == 0) dataset.GenerateBlobs();
        dataset.ReseedAndGenerateClean();

        var X = dataset.XMatrix(); var Y = dataset.YMatrix();
        int N = dataset.count, Nval = Mathf.RoundToInt(N * valSplit), Ntr = N - Nval;

        int[] idx = new int[N]; for (int i = 0; i < N; i++) idx[i] = i;
        for (int i = 0; i < N; i++) { int j = rnd.Next(N); (idx[i], idx[j]) = (idx[j], idx[i]); }

        Xtr = new float[Ntr, 2]; Ytr = new float[Ntr, 1];
        Xval = new float[Nval, 2]; Yval = new float[Nval, 1];
        for (int k = 0; k < Ntr; k++) { int i = idx[k]; Xtr[k, 0] = X[i, 0]; Xtr[k, 1] = X[i, 1]; Ytr[k, 0] = Y[i, 0]; }
        for (int k = 0; k < Nval; k++) { int i = idx[Ntr + k]; Xval[k, 0] = X[i, 0]; Xval[k, 1] = X[i, 1]; Yval[k, 0] = Y[i, 0]; }
    }

    void ConfigureBounds()
    {
        // derive world bounds from dataset with a small margin
        float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
        foreach (var p in dataset.points) { if (p.x < minx) minx = p.x; if (p.y < miny) miny = p.y; if (p.x > maxx) maxx = p.x; if (p.y > maxy) maxy = p.y; }
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
        btnStep.onClick.AddListener(Step);
        tglAuto.onValueChanged.AddListener(v => auto = v);
        sldLR.onValueChanged.AddListener(v => mlp.lr = v);
        btnRebuild.onClick.AddListener(() => { BuildModel(); RedrawAll(); });

        sldL1.onValueChanged.AddListener(_ => { mlp.l1 = sldL1.value; RedrawAll(); });
        sldL2.onValueChanged.AddListener(_ => { mlp.l2 = sldL2.value; RedrawAll(); });
        sldDrop.onValueChanged.AddListener(_ => { mlp.dropoutP = sldDrop.value; RedrawAll(); });

        btnReset.onClick.AddListener(() => { BuildModel(); RedrawAll(); });
        btnShuffle.onClick.AddListener(() => { InitData(); ConfigureBounds(); RedrawAll(); });
    }

    void Update()
    {
        if (!auto) return;
        t += Time.deltaTime; if (t >= dt) { t = 0f; Step(); }
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
        else { mlp.Forward(Xtr, Ytr, train: true); mlp.StepSGD(N); }
        RedrawAll();
    }

    void RedrawAll()
    {
        var (ltr, Ptr) = mlp.Forward(Xtr, Ytr, train: false);
        var (lva, Pva) = mlp.Forward(Xval, Yval, train: false);

        float atr = Accuracy(Ptr, Ytr, 0.5f);
        float ava = Accuracy(Pva, Yval, 0.5f);
        float gap = Mathf.Max(0f, atr - ava);
        float score = ava - 0.5f * gap;

        if (txtTrain) txtTrain.text = $"Train: acc {atr:0.000} | loss {ltr:0.0000}";
        if (txtVal) txtVal.text = $"Val:   acc {ava:0.000} | loss {lva:0.0000}";
        if (txtScore) txtScore.text = $"Generalization Score: {score:0.000}";

        skyline?.Redraw(mlp);
        dropoutRain?.Redraw(mlp);
        gapGauge?.Redraw(gap);

        float sp = PercentZeroWeights(mlp);
        if (txtSparsity) txtSparsity.text = $"Sparsity: {sp:0.0}%";

        // NEW: decision contour + error highlights
        decision?.Redraw(mlp);
        if (highlightMistakes) UpdateDotErrors();
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

            var baseCol = isPos ? new Color(0.94f, 0.45f, 0.45f, 0.75f)
                                : new Color(0.45f, 0.62f, 0.94f, 0.75f);

            if (!ok)
            {
                dotSR[i].color = mistakeColor;        // highlight misclassified
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
    { int N = P.GetLength(0), ok = 0; for (int i = 0; i < N; i++) { int h = P[i, 0] >= thr ? 1 : 0; int y = Y[i, 0] > 0.5f ? 1 : 0; if (h == y) ok++; } return ok / (float)Mathf.Max(1, N); }

    float PercentZeroWeights(MLP_Capacity m)
    {
        int z = 0, n = 0;
        foreach (var L in m.Ls)
            for (int i = 0; i < L.W.GetLength(0); i++)
                for (int j = 0; j < L.W.GetLength(1); j++) { n++; if (Mathf.Approximately(L.W[i, j], 0f)) z++; }
        return n > 0 ? (100f * z) / n : 0f;
    }

    void SpawnDots()
    {
        for (int i = pointsParent.childCount - 1; i >= 0; i--) Destroy(pointsParent.GetChild(i).gameObject);
        dotSR.Clear();
        for (int i = 0; i < dataset.points.Length; i++)
        {
            var go = Instantiate(pointPrefab, dataset.points[i], Quaternion.identity, pointsParent);
            var sr = go.GetComponent<SpriteRenderer>();
            sr.sortingOrder = 10;
            sr.color = dataset.labels[i] > 0.5f ? new Color(0.94f, 0.45f, 0.45f, 0.75f)
                                            : new Color(0.45f, 0.62f, 0.94f, 0.75f);
            go.transform.localScale = Vector3.one * normalDotScale;
            dotSR.Add(sr);
        }
    }

    int[] SampleBatch(int bs, int total)
    {
        var set = new HashSet<int>(); while (set.Count < bs) set.Add(rnd.Next(total));
        var arr = new int[bs]; int k = 0; foreach (var i in set) arr[k++] = i; return arr;
    }

    (float[,], float[,]) GatherBatch(int[] idx, float[,] X, float[,] Y)
    {
        var Xb = new float[idx.Length, 2]; var Yb = new float[idx.Length, 1];
        for (int r = 0; r < idx.Length; r++) { int i = idx[r]; Xb[r, 0] = X[i, 0]; Xb[r, 1] = X[i, 1]; Yb[r, 0] = Y[i, 0]; }
        return (Xb, Yb);
    }
}
