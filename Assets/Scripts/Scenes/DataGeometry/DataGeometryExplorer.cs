using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;

public class DataGeometryExplorer : MonoBehaviour
{
    [Header("World")]
    public Transform pointsParent;
    public GameObject pointPrefab;
    public DecisionContourPanel decision;

    [Header("Dataset Controls")]
    public TMP_Dropdown drpDataset;        // Blobs, Moons, Rings
    public Slider sldNoise, sldOverlap, sldRot;
    public Button btnRespawn;

    [Header("Model Controls")]
    public Slider sldUnits, sldLR, sldBatch;
    public TMP_Dropdown drpLayers;
    public Toggle tglAuto, tglMinibatch;
    public Button btnStep, btnResetNet;

    [Header("Meters")]
    public AmbiguityMeterPanel ambiguity;
    public ComplexityMeterPanel complexity;
    public TMP_Text txtTrain, txtVal;

    [Header("Look & Feel")]
    public bool highlightMistakes = true;
    public Color mistakeColor = new(1f, 0.85f, 0.2f, 1f);
    [Range(0.05f, 0.2f)] public float normalDotScale = 0.09f;
    [Range(0.06f, 0.25f)] public float mistakeDotScale = 0.13f;

    // data
    Vector2[] pts; int[] labels;
    float[,] Xtr, Ytr, Xval, Yval;
    readonly List<SpriteRenderer> dotSR = new();

    // model
    MLP_Capacity mlp;
    bool auto = false; float accTrain = 0f, accVal = 0f;
    System.Random rnd = new System.Random(42);

    void Start()
    {
        BuildData();
        BuildModel();
        WireUI();
        SpawnDots();
        ConfigureBounds();
        RedrawAll();
    }

    void WireUI()
    {
        btnRespawn.onClick.AddListener(() => { BuildData(); SpawnDots(); ConfigureBounds(); RedrawAll(); });
        btnResetNet.onClick.AddListener(() => { BuildModel(); RedrawAll(); });
        drpDataset.onValueChanged.AddListener(_ => { BuildData(); SpawnDots(); ConfigureBounds(); RedrawAll(); });
        sldNoise.onValueChanged.AddListener(_ => { BuildData(); SpawnDots(); ConfigureBounds(); RedrawAll(); });
        sldOverlap.onValueChanged.AddListener(_ => { BuildData(); SpawnDots(); ConfigureBounds(); RedrawAll(); });
        sldRot.onValueChanged.AddListener(_ => { BuildData(); SpawnDots(); ConfigureBounds(); RedrawAll(); });

        btnStep.onClick.AddListener(Step);
        tglAuto.onValueChanged.AddListener(v => auto = v);
        sldLR.onValueChanged.AddListener(v => mlp.lr = v);
    }

    void Update() { if (auto) { Step(); } }

    void Step()
    {
        if (tglMinibatch && tglMinibatch.isOn)
        {
            int N = Xtr.GetLength(0);
            int bs = Mathf.Clamp((int)sldBatch.value, 1, N);
            var idx = SampleBatch(bs, N);
            var (Xb, Yb) = GatherBatch(idx, Xtr, Ytr);
            mlp.Forward(Xb, Yb, train: true);
            mlp.StepSGD(bs);
        }
        else
        {
            mlp.Forward(Xtr, Ytr, train: true);
            mlp.StepSGD(Xtr.GetLength(0));
        }
        RedrawAll();
    }

    void RedrawAll()
    {
        var (ltr, Ptr) = mlp.Forward(Xtr, Ytr, train: false);
        var (lva, Pva) = mlp.Forward(Xval, Yval, train: false);

        accTrain = Accuracy(Ptr, Ytr, 0.5f);
        accVal = Accuracy(Pva, Yval, 0.5f);
        if (txtTrain) txtTrain.text = $"Train acc {accTrain:0.000} | loss {ltr:0.0000}";
        if (txtVal) txtVal.text = $"Val   acc {accVal:0.000} | loss {lva:0.0000}";

        decision?.Redraw(mlp);
        ambiguity?.Redraw(pts, labels);
        complexity?.Redraw(mlp);

        if (highlightMistakes) UpdateDotErrors();
    }

    void BuildModel()
    {
        int H = Mathf.Max(2, (int)sldUnits.value);
        int L = Mathf.Clamp(drpLayers.value + 1, 1, 3);
        mlp = new MLP_Capacity(2, H, 1, layers: L, seed: UnityEngine.Random.Range(1, 1_000_000))
        {
            activation = MLP_Capacity.Act.ReLU,
            lossType = MLP_Capacity.LossType.BCE,
            lr = sldLR.value,
            l1 = 0f,
            l2 = 0.001f,
            dropoutP = 0f
        };
    }

    void BuildData()
    {
        int n = 600;
        int typ = drpDataset ? drpDataset.value : 0;
        float noise = sldNoise ? sldNoise.value : 0.08f;
        float ov = sldOverlap ? sldOverlap.value : 0.4f;
        float rot = sldRot ? sldRot.value : 0f;

        switch (typ)
        {
            case 0: DataFactory2D.MakeBlobs(n, noise, ov, rot, out pts, out labels); break;
            case 1: DataFactory2D.MakeMoons(n, noise, ov, rot, out pts, out labels); break;
            case 2: DataFactory2D.MakeRings(n, noise, ov, rot, out pts, out labels); break;
            default: DataFactory2D.MakeBlobs(n, noise, ov, rot, out pts, out labels); break;
        }

        // train/val split
        int N = pts.Length, Nval = Mathf.RoundToInt(0.25f * N), Ntr = N - Nval;
        int[] idx = new int[N]; for (int i = 0; i < N; i++) idx[i] = i;
        for (int i = 0; i < N; i++) { int j = rnd.Next(N); (idx[i], idx[j]) = (idx[j], idx[i]); }

        Xtr = new float[Ntr, 2]; Ytr = new float[Ntr, 1];
        Xval = new float[Nval, 2]; Yval = new float[Nval, 1];
        for (int k = 0; k < Ntr; k++) { int i = idx[k]; Xtr[k, 0] = pts[i].x; Xtr[k, 1] = pts[i].y; Ytr[k, 0] = labels[i]; }
        for (int k = 0; k < Nval; k++) { int i = idx[Ntr + k]; Xval[k, 0] = pts[i].x; Xval[k, 1] = pts[i].y; Yval[k, 0] = labels[i]; }
    }

    void ConfigureBounds()
    {
        float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
        foreach (var p in pts) { if (p.x < minx) minx = p.x; if (p.y < miny) miny = p.y; if (p.x > maxx) maxx = p.x; if (p.y > maxy) maxy = p.y; }
        Vector2 size = new(maxx - minx, maxy - miny); Vector2 pad = 0.15f * size;
        var wmin = new Vector2(minx, miny) - pad; var wmax = new Vector2(maxx, maxy) + pad;
        decision?.Configure(wmin, wmax);
        complexity?.Configure(wmin, wmax);
    }

    void SpawnDots()
    {
        for (int i = pointsParent.childCount - 1; i >= 0; i--) Destroy(pointsParent.GetChild(i).gameObject);
        dotSR.Clear();
        for (int i = 0; i < pts.Length; i++)
        {
            var go = Instantiate(pointPrefab, pts[i], Quaternion.identity, pointsParent);
            var sr = go.GetComponent<SpriteRenderer>(); sr.sortingOrder = 10;
            sr.color = labels[i] == 1 ? new Color(0.94f, 0.45f, 0.45f, 0.75f) : new Color(0.45f, 0.62f, 0.94f, 0.75f);
            go.transform.localScale = Vector3.one * normalDotScale;
            dotSR.Add(sr);
        }
    }

    void UpdateDotErrors()
    {
        var P = mlp.Forward(ToX(pts), null, train: false).pred;
        for (int i = 0; i < dotSR.Count; i++)
        {
            bool isPos = labels[i] == 1;
            bool predPos = P[i, 0] >= 0.5f;
            bool ok = (isPos == predPos);
            if (!ok) { dotSR[i].color = mistakeColor; dotSR[i].transform.localScale = Vector3.one * mistakeDotScale; }
            else
            {
                dotSR[i].color = isPos ? new Color(0.94f, 0.45f, 0.45f, 0.75f) : new Color(0.45f, 0.62f, 0.94f, 0.75f);
                dotSR[i].transform.localScale = Vector3.one * normalDotScale;
            }
        }
    }

    // utils
    float[,] ToX(Vector2[] p) { var X = new float[p.Length, 2]; for (int i = 0; i < p.Length; i++) { X[i, 0] = p[i].x; X[i, 1] = p[i].y; } return X; }
    float Accuracy(float[,] P, float[,] Y, float thr) { int N = P.GetLength(0), ok = 0; for (int i = 0; i < N; i++) { int h = P[i, 0] >= thr ? 1 : 0; int y = (Y[i, 0] > 0.5f) ? 1 : 0; if (h == y) ok++; } return ok / (float)Mathf.Max(1, N); }
    int[] SampleBatch(int bs, int total) { var set = new HashSet<int>(); while (set.Count < bs) set.Add(rnd.Next(total)); var arr = new int[bs]; int k = 0; foreach (var i in set) arr[k++] = i; return arr; }
    (float[,], float[,]) GatherBatch(int[] idx, float[,] X, float[,] Y)
    {
        var Xb = new float[idx.Length, 2]; var Yb = new float[idx.Length, 1];
        for (int r = 0; r < idx.Length; r++) { int i = idx[r]; Xb[r, 0] = X[i, 0]; Xb[r, 1] = X[i, 1]; Yb[r, 0] = Y[i, 0]; }
        return (Xb, Yb);
    }
}
