using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System;
using System.Collections.Generic;

/// Scene 7 controller with stronger visuals + dataset randomization on Respawn/Rebuild.
public class AttributionExplorer : MonoBehaviour
{
    [Header("Data")]
    public Dataset2D_S6 dataset;
    [Range(0.05f, 0.5f)] public float valSplit = 0.25f;

    [Header("Dataset Defaults (used for Respawn/Rebuild)")]
    public Dataset2D_S6.Shape defaultShape = Dataset2D_S6.Shape.Moons;
    [Range(100, 5000)] public int defaultSamples = 800;
    [Range(0f, 0.8f)] public float defaultNoise = 0.08f;
    [Range(0f, 1f)] public float defaultOverlap = 0.25f;

    [Header("World")]
    public Transform pointsParent;
    public GameObject pointPrefab;             // small SpriteRenderer dot
    public DecisionContourPanel decision;      // from Scene 5
    public LineRenderer surrogateLine;         // dashed/solid line
    public LineRenderer gradArrow;             // little arrow at point
    public SpriteRenderer counterfactualGhost; // small circle sprite

    [Header("Train")]
    public Slider sldLR, sldBatch; public Toggle tglAuto, tglMinibatch;
    public Button btnStep, btnReset, btnRebuild, btnRespawn; // ← Respawn added
    public TMP_Text txtTrain, txtVal, txtScore;

    [Header("Attribution Controls")]
    public Slider sldIGSteps;     // 8..128
    public Slider sldLimeSigma;   // 0.05..0.5
    public Slider sldCFSteps;     // 1..12
    public Slider sldCFAlpha;     // 0.2..2.0
    public Toggle tglShowWind, tglShowSur, tglShowIG, tglShowCF;

    [Header("Panels")]
    public IGBarsPanel igBars;
    public GradientFieldPanel gradField;       // optional

    [Header("Style (visibility)")]
    [Range(0.06f, 0.30f)] public float dotScale = 0.14f;
    [Range(0.08f, 0.40f)] public float selectedDotScale = 0.20f;
    [Range(0.010f, 0.070f)] public float lineWidth = 0.035f;
    [Tooltip("Arrow length as a fraction of scene world size")]
    [Range(0.05f, 0.60f)] public float arrowLength = 0.28f;
    public Color colorPos = new Color(1.00f, 0.47f, 0.47f, 0.95f);
    public Color colorNeg = new Color(0.38f, 0.67f, 1.00f, 0.95f);
    public Color colorGhost = new Color(1f, 1f, 1f, 0.95f);

    // internals
    MLP_Capacity mlp;
    float[,] Xtr, Ytr, Xval, Yval;
    readonly List<SpriteRenderer> dots = new List<SpriteRenderer>();
    System.Random rnd = new System.Random(7);

    bool auto = false; float tick = 0f; const float dt = 0.05f;
    int selIndex = -1; Vector2 selPoint;
    Vector2 wmin, wmax;

    void Start()
    {
        if (!dataset) dataset = ScriptableObject.CreateInstance<Dataset2D_S6>();
        // initial data
        GenerateDatasetWithNewSeed();

        SplitTrainVal();
        BuildModel();
        WireUI();
        SpawnDots();
        ConfigureBounds();
        ApplyStyleToLines();
        RedrawAll();
    }

    void WireUI()
    {
        btnStep.onClick.AddListener(Step);
        tglAuto.onValueChanged.AddListener(v => auto = v);
        sldLR.onValueChanged.AddListener(v => mlp.lr = v);

        // Reset = rebuild model only (keep current data)
        if (btnReset) btnReset.onClick.AddListener(() => { BuildModel(); RedrawAll(); });

        // Respawn = randomize dataset, keep current model weights
        if (btnRespawn) btnRespawn.onClick.AddListener(RandomizeDataset);

        // Rebuild = randomize dataset AND rebuild model
        if (btnRebuild) btnRebuild.onClick.AddListener(RandomizeAndRebuild);

        if (tglShowWind) tglShowWind.onValueChanged.AddListener(_ => RedrawAll());
        if (tglShowSur) tglShowSur.onValueChanged.AddListener(_ => RedrawAll());
        if (tglShowIG) tglShowIG.onValueChanged.AddListener(_ => RedrawAll());
        if (tglShowCF) tglShowCF.onValueChanged.AddListener(_ => RedrawAll());
    }

    // ---------- Dataset ops ----------
    void GenerateDatasetWithNewSeed()
    {
        dataset.Generate(defaultShape, defaultSamples, defaultNoise, defaultOverlap,
                         UnityEngine.Random.Range(1, 1_000_000));
    }

    public void RandomizeDataset()
    {
        GenerateDatasetWithNewSeed();
        SplitTrainVal();
        SpawnDots();
        ConfigureBounds();
        RedrawAll();
    }

    public void RandomizeAndRebuild()
    {
        GenerateDatasetWithNewSeed();
        SplitTrainVal();
        SpawnDots();
        ConfigureBounds();
        BuildModel();
        RedrawAll();
    }

    void Update()
    {
        if (Input.GetMouseButtonDown(0)) SelectNearestPointAtMouse();
        if (!auto) return;
        tick += Time.deltaTime;
        if (tick >= dt) { tick = 0f; Step(); }
    }

    void SelectNearestPointAtMouse()
    {
        Vector3 w = Camera.main.ScreenToWorldPoint(Input.mousePosition); w.z = 0f;
        int best = -1; float bestD = 1e9f;
        for (int i = 0; i < dataset.count; i++)
        {
            float d = (dataset.points[i] - (Vector2)w).sqrMagnitude;
            if (d < bestD) { bestD = d; best = i; }
        }
        if (best >= 0) { selIndex = best; selPoint = dataset.points[best]; EmphasizeSelection(); RedrawAttribution(); }
    }

    void EmphasizeSelection()
    {
        for (int i = 0; i < dots.Count; i++)
        {
            float s = (i == selIndex) ? selectedDotScale : dotScale;
            dots[i].transform.localScale = Vector3.one * s;
        }
    }

    void SplitTrainVal()
    {
        int N = dataset.count, Nv = Mathf.RoundToInt(N * valSplit), Nt = N - Nv;
        int[] idx = new int[N]; for (int i = 0; i < N; i++) idx[i] = i;
        for (int i = 0; i < N; i++) { int j = rnd.Next(N); (idx[i], idx[j]) = (idx[j], idx[i]); }

        Xtr = new float[Nt, 2]; Ytr = new float[Nt, 1]; Xval = new float[Nv, 2]; Yval = new float[Nv, 1];
        for (int k = 0; k < Nt; k++) { int i = idx[k]; Xtr[k, 0] = dataset.points[i].x; Xtr[k, 1] = dataset.points[i].y; Ytr[k, 0] = dataset.labels[i]; }
        for (int k = 0; k < Nv; k++) { int i = idx[Nt + k]; Xval[k, 0] = dataset.points[i].x; Xval[k, 1] = dataset.points[i].y; Yval[k, 0] = dataset.labels[i]; }
    }

    void BuildModel()
    {
        mlp = new MLP_Capacity(2, hidden: 16, output: 1, layers: 2, seed: UnityEngine.Random.Range(1, 1_000_000))
        {
            lr = sldLR ? sldLR.value : 0.1f,
            lossType = MLP_Capacity.LossType.BCE,
            activation = MLP_Capacity.Act.ReLU,
            l1 = 0f,
            l2 = 0.001f,
            dropoutP = 0f
        };
    }

    void SpawnDots()
    {
        for (int i = pointsParent.childCount - 1; i >= 0; i--) Destroy(pointsParent.GetChild(i).gameObject);
        dots.Clear();

        for (int i = 0; i < dataset.count; i++)
        {
            var go = Instantiate(pointPrefab, dataset.points[i], Quaternion.identity, pointsParent);
            var sr = go.GetComponent<SpriteRenderer>(); sr.sortingOrder = 10;
            sr.color = dataset.labels[i] > 0.5f ? colorPos : colorNeg;
            go.transform.localScale = Vector3.one * dotScale;
            dots.Add(sr);
        }
    }

    void ConfigureBounds()
    {
        float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
        foreach (var p in dataset.points) { if (p.x < minx) minx = p.x; if (p.y < miny) miny = p.y; if (p.x > maxx) maxx = p.x; if (p.y > maxy) maxy = p.y; }
        Vector2 size = new Vector2(maxx - minx, maxy - miny);
        Vector2 pad = 0.15f * size;
        wmin = new Vector2(minx, miny) - pad;
        wmax = new Vector2(maxx, maxy) + pad;
        if (decision) decision.Configure(wmin, wmax);
        if (gradField) gradField.Configure(wmin, wmax);
    }

    void ApplyStyleToLines()
    {
        if (surrogateLine)
        {
            surrogateLine.widthMultiplier = lineWidth;
            surrogateLine.numCapVertices = 8; surrogateLine.numCornerVertices = 8;
        }
        if (gradArrow)
        {
            gradArrow.widthMultiplier = lineWidth;
            gradArrow.numCapVertices = 8; gradArrow.numCornerVertices = 8;
            gradArrow.startColor = Color.white; gradArrow.endColor = Color.white;
        }
        if (counterfactualGhost)
        {
            counterfactualGhost.color = colorGhost;
            counterfactualGhost.transform.localScale = Vector3.one * (selectedDotScale * 0.9f);
        }
    }

    void Step()
    {
        int N = Xtr.GetLength(0);
        if (tglMinibatch && tglMinibatch.isOn)
        {
            int bs = Mathf.Clamp((int)sldBatch.value, 1, N);
            var idx = SampleBatch(bs, N);
            var (Xb, Yb) = Gather(idx, Xtr, Ytr);
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
        float atr = Acc(Ptr, Ytr, 0.5f), ava = Acc(Pva, Yval, 0.5f);
        float gap = Mathf.Max(0f, atr - ava); float score = ava - 0.5f * gap;

        if (txtTrain) txtTrain.text = $"TRAIN:\nACC {atr:0.000}\n| LOSS\n{ltr:0.0000}";
        if (txtVal) txtVal.text = $"VAL:\nACC {ava:0.000}\n| LOSS\n{lva:0.0000}";
        if (txtScore) txtScore.text = $"GENERALIZ-\nATION\nSCORE:\n{score:0.000}";

        if (decision) decision.Redraw(mlp);
        if (tglShowWind && tglShowWind.isOn) { if (gradField) gradField.Redraw(mlp, 12); }
        else { if (gradField) gradField.Clear(); }

        if (selIndex >= 0) RedrawAttribution();
    }

    void RedrawAttribution()
    {
        Vector2 g = GradP(selPoint);
        DrawGradArrow(selPoint, g);

        if (tglShowSur && tglShowSur.isOn) DrawLocalSurrogate(selPoint, sldLimeSigma ? sldLimeSigma.value : 0.15f);
        else if (surrogateLine) surrogateLine.positionCount = 0;

        if (tglShowIG && tglShowIG.isOn)
        {
            int m = sldIGSteps ? Mathf.Clamp((int)sldIGSteps.value, 8, 256) : 32;
            Vector2 baseline = Vector2.zero;
            Vector2 ig = IntegratedGradients(selPoint, baseline, m);
            if (igBars) igBars.Redraw(ig.x, ig.y);
        }
        else if (igBars) igBars.Clear();

        if (tglShowCF && tglShowCF.isOn)
        {
            int steps = sldCFSteps ? Mathf.Clamp((int)sldCFSteps.value, 1, 20) : 6;
            float alpha = sldCFAlpha ? Mathf.Clamp(sldCFAlpha.value, 0.05f, 4f) : 0.8f;
            Vector2 cf = CounterfactualToBoundary(selPoint, steps, alpha);
            DrawCounterfactual(selPoint, cf);
        }
        else
        {
            if (counterfactualGhost) counterfactualGhost.enabled = false;
        }
    }

    // ---- Attribution primitives ----

    Vector2 GradP(Vector2 x)
    {
        var X = new float[1, 2]; X[0, 0] = x.x; X[0, 1] = x.y;
        var (_, P) = mlp.Forward(X, null, train: false);
        float p = Mathf.Clamp01(P[0, 0]);
        float dpdz = p * (1f - p);

        int Llast = mlp.Ls.Length - 1;
        var Lout = mlp.Ls[Llast];
        int H = Lout.W.GetLength(0);
        float[] gA = new float[H];
        for (int j = 0; j < H; j++) gA[j] = dpdz * Lout.W[j, 0];

        for (int l = Llast - 1; l >= 0; l--)
        {
            var L = mlp.Ls[l];
            int outDim = L.W.GetLength(1);
            int inDim = L.W.GetLength(0);

            float[] actp = new float[outDim];
            for (int j = 0; j < outDim; j++)
            {
                float z = L.Z[0, j];
                if (mlp.activation == MLP_Capacity.Act.ReLU) actp[j] = z > 0f ? 1f : 0f;
                else { float a = L.A[0, j]; actp[j] = 1f - a * a; }
            }

            float[] gZ = new float[outDim];
            for (int j = 0; j < outDim; j++) gZ[j] = gA[j] * actp[j];

            float[] gAprev = new float[inDim];
            for (int i = 0; i < inDim; i++)
                for (int j = 0; j < outDim; j++)
                    gAprev[i] += gZ[j] * L.W[i, j];

            gA = gAprev;
            if (l == 0) return new Vector2(gA[0], gA[1]);
        }
        return Vector2.zero;
    }

    Vector2 IntegratedGradients(Vector2 x, Vector2 baseline, int mSteps)
    {
        Vector2 accum = Vector2.zero;
        for (int k = 1; k <= mSteps; k++)
        {
            float t = k / (float)mSteps;
            Vector2 xt = baseline + t * (x - baseline);
            Vector2 g = GradP(xt);
            accum += g;
        }
        Vector2 avgGrad = accum / Mathf.Max(1, mSteps);
        return new Vector2((x.x - baseline.x) * avgGrad.x, (x.y - baseline.y) * avgGrad.y);
    }

    void DrawGradArrow(Vector2 x, Vector2 g)
    {
        if (!gradArrow) return;
        float len = g.magnitude; if (len < 1e-6f) { gradArrow.positionCount = 0; return; }
        Vector2 dir = g / len;
        float worldSize = Mathf.Max(wmax.x - wmin.x, wmax.y - wmin.y);
        float scale = arrowLength * worldSize;
        Vector3 a = new Vector3(x.x, x.y, 0f);
        Vector3 b = new Vector3(x.x + dir.x * scale, x.y + dir.y * scale, 0f);
        gradArrow.positionCount = 2;
        gradArrow.SetPosition(0, a);
        gradArrow.SetPosition(1, b);
    }

    void DrawLocalSurrogate(Vector2 x, float sigma)
    {
        int K = 128;
        double S00 = 0, S01 = 0, S02 = 0, S11 = 0, S12 = 0, S22 = 0;
        double T0 = 0, T1 = 0, T2 = 0;
        for (int i = 0; i < K; i++)
        {
            Vector2 xi = x + sigma * RandN2();
            float[,] Xin = new float[1, 2]; Xin[0, 0] = xi.x; Xin[0, 1] = xi.y;
            float p = mlp.Forward(Xin, null, train: false).pred[0, 0];
            double w = Math.Exp(-((xi - x).sqrMagnitude) / (2.0 * sigma * sigma));
            double z0 = 1.0, z1 = xi.x, z2 = xi.y;
            S00 += w * z0 * z0; S01 += w * z0 * z1; S02 += w * z0 * z2;
            S11 += w * z1 * z1; S12 += w * z1 * z2; S22 += w * z2 * z2;
            T0 += w * z0 * p; T1 += w * z1 * p; T2 += w * z2 * p;
        }
        double[,] S = { { S00, S01, S02 }, { S01, S11, S12 }, { S02, S12, S22 } };
        double[] T = { T0, T1, T2 };
        double[] beta = Solve3x3(S, T); // [b0, bx, by]

        if (!surrogateLine) { return; }
        double bx = beta[1], by = beta[2], b0 = beta[0] - 0.5;
        Vector2 pA, pB; bool ok = LineWithinBounds((float)bx, (float)by, (float)b0, out pA, out pB);
        if (ok)
        {
            surrogateLine.widthMultiplier = lineWidth;
            surrogateLine.positionCount = 2;
            surrogateLine.SetPosition(0, new Vector3(pA.x, pA.y, 0f));
            surrogateLine.SetPosition(1, new Vector3(pB.x, pB.y, 0f));
        }
        else surrogateLine.positionCount = 0;
    }

    Vector2 CounterfactualToBoundary(Vector2 x0, int steps, float alpha)
    {
        Vector2 x = x0;
        for (int t = 0; t < steps; t++)
        {
            float p = mlp.Forward(new float[,] { { x.x, x.y } }, null, false).pred[0, 0];
            Vector2 g = GradP(x);
            float g2 = g.sqrMagnitude + 1e-8f;
            x -= alpha * ((p - 0.5f) / g2) * g;
            x = new Vector2(Mathf.Clamp(x.x, wmin.x, wmax.x), Mathf.Clamp(x.y, wmin.y, wmax.y));
        }
        return x;
    }

    void DrawCounterfactual(Vector2 x, Vector2 cf)
    {
        if (!counterfactualGhost) return;
        counterfactualGhost.enabled = true;
        counterfactualGhost.transform.position = new Vector3(cf.x, cf.y, 0f);
        counterfactualGhost.transform.localScale = Vector3.one * (selectedDotScale * 0.9f);
    }

    // ---- math/util ----
    static Vector2 RandN2()
    {
        float u = UnityEngine.Random.value, v = UnityEngine.Random.value;
        float r = Mathf.Sqrt(-2f * Mathf.Log(1 - u)); float th = 2 * Mathf.PI * v;
        return new Vector2(r * Mathf.Cos(th), r * Mathf.Sin(th));
    }

    double[] Solve3x3(double[,] A, double[] b)
    {
        double a = A[0, 0], b01 = A[0, 1], c = A[0, 2];
        double d = A[1, 0], e = A[1, 1], f = A[1, 2];
        double g = A[2, 0], h = A[2, 1], i = A[2, 2];
        double det = a * (e * i - f * h) - b01 * (d * i - f * g) + c * (d * h - e * g);
        if (Math.Abs(det) < 1e-9) return new double[] { 0, 0, 0 };
        double invDet = 1.0 / det;
        double[,] inv = new double[3, 3];
        inv[0, 0] = (e * i - f * h) * invDet; inv[0, 1] = -(b01 * i - c * h) * invDet; inv[0, 2] = (b01 * f - c * e) * invDet;
        inv[1, 0] = -(d * i - f * g) * invDet; inv[1, 1] = (a * i - c * g) * invDet; inv[1, 2] = -(a * f - c * d) * invDet;
        inv[2, 0] = (d * h - e * g) * invDet; inv[2, 1] = -(a * h - b01 * g) * invDet; inv[2, 2] = (a * e - b01 * d) * invDet;
        double[] x = new double[3];
        for (int r = 0; r < 3; r++) { x[r] = inv[r, 0] * b[0] + inv[r, 1] * b[1] + inv[r, 2] * b[2]; }
        return x;
    }

    bool LineWithinBounds(float bx, float by, float b0, out Vector2 pA, out Vector2 pB)
    {
        pA = Vector2.zero; pB = Vector2.zero; int hit = 0;
        if (Mathf.Abs(by) > 1e-6f) { float y = (-(b0 + bx * wmin.x)) / by; if (y >= wmin.y && y <= wmax.y) { if (hit == 0) pA = new Vector2(wmin.x, y); else pB = new Vector2(wmin.x, y); hit++; } }
        if (Mathf.Abs(by) > 1e-6f) { float y = (-(b0 + bx * wmax.x)) / by; if (y >= wmin.y && y <= wmax.y) { if (hit == 0) pA = new Vector2(wmax.x, y); else pB = new Vector2(wmax.x, y); hit++; } }
        if (Mathf.Abs(bx) > 1e-6f) { float x = (-(b0 + by * wmin.y)) / bx; if (x >= wmin.x && x <= wmax.x) { if (hit == 0) pA = new Vector2(x, wmin.y); else pB = new Vector2(x, wmin.y); hit++; } }
        if (Mathf.Abs(bx) > 1e-6f) { float x = (-(b0 + by * wmax.y)) / bx; if (x >= wmin.x && x <= wmax.x) { if (hit == 0) pA = new Vector2(x, wmax.y); else pB = new Vector2(x, wmax.y); hit++; } }
        return hit >= 2;
    }

    float Acc(float[,] P, float[,] Y, float thr) { int N = P.GetLength(0), ok = 0; for (int i = 0; i < N; i++) { int h = P[i, 0] >= thr ? 1 : 0; int y = Y[i, 0] > 0.5f ? 1 : 0; if (h == y) ok++; } return ok / (float)Mathf.Max(1, N); }
    int[] SampleBatch(int bs, int total) { var set = new HashSet<int>(); while (set.Count < bs) set.Add(rnd.Next(total)); var arr = new int[bs]; int k = 0; foreach (var i in set) arr[k++] = i; return arr; }
    (float[,], float[,]) Gather(int[] idx, float[,] X, float[,] Y) { var Xb = new float[idx.Length, 2]; var Yb = new float[idx.Length, 1]; for (int r = 0; r < idx.Length; r++) { int i = idx[r]; Xb[r, 0] = X[i, 0]; Xb[r, 1] = X[i, 1]; Yb[r, 0] = Y[i, 0]; } return (Xb, Yb); }
}
