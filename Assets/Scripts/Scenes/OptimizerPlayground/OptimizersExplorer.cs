using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System;
using System.Collections.Generic;

public class OptimizersExplorer : MonoBehaviour
{
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

    void Start()
    {
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
        mlpSGD = new MLP(2, 3, 1, seed) { lossType = LossType.BCE, activation = (Act)drpActivation.value };
        mlpMom = CloneMLP(mlpSGD);
        mlpAdam = CloneMLP(mlpSGD);

        // optimizer params
        optMom.beta = sldMomentum.value;
        optAdam.beta1 = sldAdamB1.value;
        optAdam.beta2 = sldAdamB2.value;

        // UI hooks
        btnStep.onClick.AddListener(Step);
        tglAuto.onValueChanged.AddListener(v => auto = v);
        drpFocus.onValueChanged.AddListener(_ => RedrawField());
        drpActivation.onValueChanged.AddListener(OnActivationChanged);
        tglMinibatch.onValueChanged.AddListener(_ => { /* no op */ });
        sldBatch.onValueChanged.AddListener(_ => { /* label optional */ });
        sldLR.onValueChanged.AddListener(_ => { /* realtime */ });
        sldMomentum.onValueChanged.AddListener(v => optMom.beta = v);
        sldAdamB1.onValueChanged.AddListener(v => optAdam.beta1 = v);
        sldAdamB2.onValueChanged.AddListener(v => optAdam.beta2 = v);
        btnReset.onClick.AddListener(ResetAll);
        btnShuffle.onClick.AddListener(ShuffleData);

        // points & field
        SpawnPoints();
        RedrawField();
        PushChart(); // initial losses
        UpdateLossLabel();
    }

    void Update()
    {
        if (!auto) return;
        timer += Time.deltaTime;
        if (timer >= dt) { timer = 0f; Step(); }
    }

    void OnActivationChanged(int idx)
    {
        mlpSGD.activation = (Act)idx;
        mlpMom.activation = (Act)idx;
        mlpAdam.activation = (Act)idx;
        RedrawField();
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

        // UI
        RedrawField();
        PushChart();
        UpdateLossLabel();
    }

    void RedrawField()
    {
        // the DecisionField shows the focused model
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

    void PushChart()
    {
        var lS = mlpSGD.Forward(X, Y).loss;
        var lM = mlpMom.Forward(X, Y).loss;
        var lA = mlpAdam.Forward(X, Y).loss;
        chart?.Push(lS, lM, lA);
    }

    void UpdateLossLabel()
    {
        if (txtLoss == null) return;
        var choice = drpFocus.value;
        float curr = (choice == 0 ? mlpSGD : choice == 1 ? mlpMom : mlpAdam).Forward(X, Y).loss;
        string name = choice == 0 ? "SGD" : choice == 1 ? "Momentum" : "Adam";
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
            sr.color = dataset.labels[i] > 0.5f ? new Color(0.94f, 0.28f, 0.28f) : new Color(0.28f, 0.46f, 0.94f);
            sr.sortingOrder = 10;
            go.transform.localScale = Vector3.one * 0.15f;
            pointGOs.Add(go);
        }
    }

    void ResetAll()
    {
        var seed = UnityEngine.Random.Range(1, 1_000_000);
        mlpSGD = new MLP(2, 3, 1, seed) { lossType = LossType.BCE, activation = (Act)drpActivation.value };
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
        PushChart();
        UpdateLossLabel();
    }

    void ShuffleData()
    {
        dataset.ReseedAndGenerateClean();
        X = dataset.XMatrix();
        Y = dataset.YMatrix();
        SpawnPoints();
        chart?.ClearSeries();
        RedrawField();
        PushChart();
        UpdateLossLabel();
    }

    int[] SampleBatch(int bs, int total)
    {
        var set = new HashSet<int>();
        while (set.Count < bs) set.Add(rnd.Next(total));
        var arr = new int[bs]; int k = 0; foreach (var i in set) arr[k++] = i; return arr;
    }
    (float[,], float[,]) GatherBatch(int[] idx)
    {
        var Xb = new float[idx.Length, 2];
        var Yb = new float[idx.Length, 1];
        for (int r = 0; r < idx.Length; r++)
        { int i = idx[r]; Xb[r, 0] = dataset.points[i].x; Xb[r, 1] = dataset.points[i].y; Yb[r, 0] = dataset.labels[i]; }
        return (Xb, Yb);
    }

    MLP CloneMLP(MLP src)
    {
        var dst = new MLP(
            src.Ls[0].W.GetLength(0), // input size
            src.Ls[0].W.GetLength(1), // hidden size
            src.Ls[1].W.GetLength(1), // output size
            seed: 1                   // seed doesn't matter; we overwrite weights
        );
        dst.lossType = src.lossType;
        dst.activation = src.activation;
        dst.lr = src.lr;

        // deep copy weights & biases (no TinyTensor.Copy needed)
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

}
