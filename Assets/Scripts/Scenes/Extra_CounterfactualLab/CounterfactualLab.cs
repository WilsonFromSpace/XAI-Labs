using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System;
using System.Collections.Generic;

/// Scene 8: Counterfactuals & Adversarial What-ifs
/// - Click a point to select it
/// - FGSM / PGD attack under L2 / L∞ / L1 with ε-ball
/// - Shows gradient-of-loss arrow, epsilon ring, adversarial path and ghost
public class CounterfactualLab : MonoBehaviour
{
    [Header("Data")]
    public Dataset2D_S6 dataset;
    [Range(0.05f, 0.5f)] public float valSplit = 0.25f;
    public Dataset2D_S6.Shape defaultShape = Dataset2D_S6.Shape.Moons;
    [Range(100, 5000)] public int defaultSamples = 800;
    [Range(0f, 0.8f)] public float defaultNoise = 0.08f;
    [Range(0f, 1f)] public float defaultOverlap = 0.25f;

    [Header("World")]
    public Transform pointsParent;
    public GameObject pointPrefab;                 // SpriteRenderer dot
    public DecisionContourPanel decision;          // re-use from earlier scenes
    public LineRenderer gradArrow;                 // gradient-of-loss at x
    public LineRenderer pathLine;                  // adversarial path
    public LineRenderer epsRing;                   // ε-ball (circle/square/diamond)
    public SpriteRenderer advGhost;                // final adversarial point
    public SpriteRenderer selHalo;                 // optional: small white halo at selected

    [Header("Training")]
    public Slider sldLR, sldBatch; public Toggle tglAuto, tglMinibatch;
    public Button btnStep, btnReset, btnRebuild, btnRespawn;
    public TMP_Text txtTrain, txtVal;

    [Header("Adversarial Controls")]
    public TMP_Dropdown drpAttack; // 0: FGSM, 1: PGD
    public TMP_Dropdown drpNorm;   // 0: L2, 1: L∞, 2: L1
    public Slider sldEps;          // 0.02..0.5 (world units)
    public Slider sldSteps;        // 1..40
    public Slider sldAlpha;        // 0.01..0.5 per step
    public Toggle tglProject;      // project back to ε-ball (PGD)
    public Toggle tglStopAtFlip;   // stop when class flips
    public Toggle tglShowArrow, tglShowRing, tglShowPath;
    public Button btnAttack, btnUndo;

    [Header("Style")]
    [Range(0.06f, 0.30f)] public float dotScale = 0.12f;
    [Range(0.08f, 0.40f)] public float selectedDotScale = 0.18f;
    [Range(0.010f, 0.070f)] public float lineWidth = 0.035f;
    [Range(0.05f, 0.60f)] public float arrowLength = 0.25f;
    public Color colPos = new Color(1.00f, 0.47f, 0.47f, 0.95f);
    public Color colNeg = new Color(0.38f, 0.67f, 1.00f, 0.95f);
    public Color colGhost = new Color(1f, 1f, 1f, 0.95f);
    public Color colRing = new Color(1f, 1f, 1f, 0.25f);
    public Color colPath = new Color(1f, 1f, 1f, 0.85f);

    // internals
    MLP_Capacity mlp;
    float[,] Xtr, Ytr, Xval, Yval;
    readonly List<SpriteRenderer> dots = new List<SpriteRenderer>();
    System.Random rnd = new System.Random(9);

    bool auto = false; float tick = 0f; const float dt = 0.05f;
    int selIndex = -1; Vector2 selPoint; float selLabel = 0f;
    Vector2 wmin, wmax;
    Vector2 advPoint; bool advActive = false;

    void Start()
    {
        if (!dataset) dataset = ScriptableObject.CreateInstance<Dataset2D_S6>();
        GenerateDatasetWithNewSeed();
        SplitTrainVal();
        BuildModel();
        WireUI();
        SpawnDots();
        ConfigureBounds();
        ApplyStyle();
        RedrawAll();
    }

    void WireUI()
    {
        if (btnStep) btnStep.onClick.AddListener(Step);
        if (tglAuto) tglAuto.onValueChanged.AddListener(v => auto = v);
        if (sldLR) sldLR.onValueChanged.AddListener(v => mlp.lr = v);

        if (btnReset) btnReset.onClick.AddListener(() => { BuildModel(); RedrawAll(); });
        if (btnRespawn) btnRespawn.onClick.AddListener(RandomizeDataset);
        if (btnRebuild) btnRebuild.onClick.AddListener(RandomizeAndRebuild);

        if (btnAttack) btnAttack.onClick.AddListener(RunAttack);
        if (btnUndo) btnUndo.onClick.AddListener(ClearAdversarial);

        if (tglShowArrow) tglShowArrow.onValueChanged.AddListener(_ => RedrawAttribution());
        if (tglShowRing) tglShowRing.onValueChanged.AddListener(_ => RedrawAttribution());
        if (tglShowPath) tglShowPath.onValueChanged.AddListener(_ => RedrawAttribution());
    }

    void Update()
    {
        if (Input.GetMouseButtonDown(0)) SelectNearestPointAtMouse();
        if (!auto) return;
        tick += Time.deltaTime;
        if (tick >= dt) { tick = 0f; Step(); }
    }

    // ---------- Dataset / model ----------
    void GenerateDatasetWithNewSeed()
    {
        dataset.Generate(defaultShape, defaultSamples, defaultNoise, defaultOverlap,
                         UnityEngine.Random.Range(1, 1_000_000));
    }
    void RandomizeDataset() { GenerateDatasetWithNewSeed(); SplitTrainVal(); SpawnDots(); ConfigureBounds(); RedrawAll(); }
    void RandomizeAndRebuild() { GenerateDatasetWithNewSeed(); SplitTrainVal(); SpawnDots(); ConfigureBounds(); BuildModel(); RedrawAll(); }

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
            sr.color = dataset.labels[i] > 0.5f ? colPos : colNeg;
            go.transform.localScale = Vector3.one * dotScale;
            dots.Add(sr);
        }
        if (selHalo) selHalo.gameObject.SetActive(false);
        if (advGhost) advGhost.enabled = false;
        if (pathLine) pathLine.positionCount = 0;
        if (epsRing) epsRing.positionCount = 0;
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
    }

    void ApplyStyle()
    {
        if (gradArrow) { gradArrow.widthMultiplier = lineWidth; gradArrow.numCapVertices = 8; gradArrow.numCornerVertices = 8; }
        if (pathLine) { pathLine.widthMultiplier = lineWidth; pathLine.numCapVertices = 8; pathLine.numCornerVertices = 8; pathLine.startColor = colPath; pathLine.endColor = colPath; }
        if (epsRing) { epsRing.widthMultiplier = lineWidth * 0.7f; epsRing.startColor = colRing; epsRing.endColor = colRing; }
        if (advGhost) { advGhost.color = colGhost; advGhost.transform.localScale = Vector3.one * (selectedDotScale * 0.9f); }
    }

    // ---------- Train / draw ----------
    void Step()
    {
        int N = Xtr.GetLength(0);
        if (tglMinibatch && tglMinibatch.isOn)
        {
            int bs = Mathf.Clamp((int)(sldBatch ? sldBatch.value : Mathf.Min(64, N)), 1, N);
            var idx = SampleBatch(bs, N);
            var g = Gather(idx, Xtr, Ytr);
            mlp.Forward(g.Item1, g.Item2, train: true);
            mlp.StepSGD(bs);
        }
        else { mlp.Forward(Xtr, Ytr, train: true); mlp.StepSGD(N); }
        RedrawAll();
    }

    void RedrawAll()
    {
        var tr = mlp.Forward(Xtr, Ytr, false);
        var va = mlp.Forward(Xval, Yval, false);
        float atr = Acc(tr.pred, Ytr, 0.5f), ava = Acc(va.pred, Yval, 0.5f);
        if (txtTrain) txtTrain.text = $"Train acc {atr:0.000} | loss {tr.loss:0.0000}";
        if (txtVal) txtVal.text = $"Val   acc {ava:0.000} | loss {va.loss:0.0000}";
        if (decision) decision.Redraw(mlp);
        if (selIndex >= 0) RedrawAttribution();
    }

    // ---------- Selection ----------
    void SelectNearestPointAtMouse()
    {
        Vector3 w = Camera.main.ScreenToWorldPoint(Input.mousePosition); w.z = 0f;
        int best = -1; float bestD = 1e9f;
        for (int i = 0; i < dataset.count; i++)
        {
            float d = (dataset.points[i] - (Vector2)w).sqrMagnitude;
            if (d < bestD) { bestD = d; best = i; }
        }
        if (best >= 0)
        {
            selIndex = best; selPoint = dataset.points[best]; selLabel = dataset.labels[best];
            EmphasizeSelection();
            ClearAdversarial();
            RedrawAttribution();
        }
    }
    void EmphasizeSelection()
    {
        for (int i = 0; i < dots.Count; i++)
        {
            float s = (i == selIndex) ? selectedDotScale : dotScale;
            dots[i].transform.localScale = Vector3.one * s;
        }
        if (selHalo)
        {
            selHalo.gameObject.SetActive(true);
            selHalo.transform.position = new Vector3(selPoint.x, selPoint.y, 0f);
            selHalo.transform.localScale = Vector3.one * (selectedDotScale * 1.2f);
        }
    }

    // ---------- Attribution primitives ----------
    // gradient of LOGIT z wrt x (2D)
    Vector2 GradLogit(Vector2 x)
    {
        var Xin = new float[1, 2]; Xin[0, 0] = x.x; Xin[0, 1] = x.y;
        var res = mlp.Forward(Xin, null, false);
        int Llast = mlp.Ls.Length - 1;
        var Lout = mlp.Ls[Llast];
        int H = Lout.W.GetLength(0);

        // start with dz/dz_out = 1 -> gA at last hidden = W[:,0]
        float[] gA = new float[H];
        for (int j = 0; j < H; j++) gA[j] = Lout.W[j, 0];

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
                else { float a = L.A[0, j]; actp[j] = 1f - a * a; } // tanh'
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

    // gradient of BCE loss wrt x: dL/dx = (p - y) * ∇x z
    Vector2 GradLoss(Vector2 x, float y)
    {
        var Xin = new float[1, 2]; Xin[0, 0] = x.x; Xin[0, 1] = x.y;
        float p = mlp.Forward(Xin, null, false).pred[0, 0];
        float coeff = p - y;
        return coeff * GradLogit(x);
    }

    // ---------- Adversarial attacks ----------
    void RunAttack()
    {
        if (selIndex < 0) return;
        int attack = drpAttack ? drpAttack.value : 0; // 0 FGSM, 1 PGD
        int norm = drpNorm ? drpNorm.value : 0; // 0 L2, 1 Linf, 2 L1
        float eps = sldEps ? sldEps.value : 0.15f;
        int steps = sldSteps ? Mathf.Max(1, (int)sldSteps.value) : 1;
        float alpha = sldAlpha ? sldAlpha.value : Mathf.Min(0.1f, eps);

        List<Vector3> path = new List<Vector3>();
        Vector2 x0 = selPoint;
        Vector2 x = x0;

        if (attack == 0) // FGSM (single step toward +loss)
        {
            Vector2 g = GradLoss(x0, selLabel);
            Vector2 d = StepDirection(g, norm);
            x = Project(x0 + eps * d, x0, eps, norm);
            path.Add(new Vector3(x0.x, x0.y, 0f));
            path.Add(new Vector3(x.x, x.y, 0f));
        }
        else // PGD (multi-step with optional projection)
        {
            path.Add(new Vector3(x0.x, x0.y, 0f));
            for (int t = 0; t < steps; t++)
            {
                Vector2 g = GradLoss(x, selLabel);
                Vector2 d = StepDirection(g, norm);
                x = x + alpha * d;                // gradient ascent on loss
                if (tglProject && tglProject.isOn) x = Project(x, x0, eps, norm);
                x = ClampWorld(x);                // keep inside scene bounds
                path.Add(new Vector3(x.x, x.y, 0f));

                // stop when class flips (optional)
                if (tglStopAtFlip && tglStopAtFlip.isOn)
                {
                    float p = mlp.Forward(new float[,] { { x.x, x.y } }, null, false).pred[0, 0];
                    int h = p >= 0.5f ? 1 : 0, y = selLabel > 0.5f ? 1 : 0;
                    if (h != y) break;
                }
            }
        }

        advPoint = x; advActive = true;
        DrawPath(path);
        DrawGhost(advPoint);
        RedrawAttribution();
    }

    Vector2 StepDirection(Vector2 g, int normKind)
    {
        // returns a unit step direction for the chosen norm (for Linf this is sign; for L2 normalize; for L1 normalize to L1=1)
        float ax = Mathf.Abs(g.x), ay = Mathf.Abs(g.y);
        if (normKind == 1)
        { // L∞
            return new Vector2(Mathf.Sign(g.x), Mathf.Sign(g.y));
        }
        else if (normKind == 0)
        { // L2
            float n = Mathf.Sqrt(g.x * g.x + g.y * g.y) + 1e-12f;
            return new Vector2(g.x / n, g.y / n);
        }
        else
        { // L1
            float n = ax + ay + 1e-12f;
            return new Vector2(g.x / n, g.y / n);
        }
    }

    Vector2 Project(Vector2 x, Vector2 center, float eps, int normKind)
    {
        Vector2 d = x - center;
        float ax = Mathf.Abs(d.x), ay = Mathf.Abs(d.y);
        if (normKind == 1)
        { // L∞: clamp box
            float px = Mathf.Clamp(d.x, -eps, eps);
            float py = Mathf.Clamp(d.y, -eps, eps);
            return center + new Vector2(px, py);
        }
        else if (normKind == 0)
        { // L2: scale to radius if outside
            float r = Mathf.Sqrt(d.x * d.x + d.y * d.y);
            if (r <= eps) return x;
            float s = eps / (r + 1e-12f);
            return center + s * d;
        }
        else
        { // L1: project to diamond (approx by scaling if outside)
            float l1 = ax + ay;
            if (l1 <= eps) return x;
            float s = eps / (l1 + 1e-12f);
            return center + s * d;
        }
    }

    Vector2 ClampWorld(Vector2 x)
    {
        return new Vector2(Mathf.Clamp(x.x, wmin.x, wmax.x), Mathf.Clamp(x.y, wmin.y, wmax.y));
    }

    void ClearAdversarial()
    {
        advActive = false;
        if (advGhost) advGhost.enabled = false;
        if (pathLine) pathLine.positionCount = 0;
        RedrawAttribution();
    }

    // ---------- Drawing ----------
    void RedrawAttribution()
    {
        DrawGradArrow();
        DrawEpsRing();
        if (!advActive) { if (pathLine) pathLine.positionCount = 0; if (advGhost) advGhost.enabled = false; }
    }

    void DrawGradArrow()
    {
        if (!gradArrow) return;
        if (!(tglShowArrow && tglShowArrow.isOn) || selIndex < 0) { gradArrow.positionCount = 0; return; }
        Vector2 g = GradLoss(selPoint, selLabel);
        float len = g.magnitude; if (len < 1e-6f) { gradArrow.positionCount = 0; return; }
        Vector2 dir = g / len;
        float worldSize = Mathf.Max(wmax.x - wmin.x, wmax.y - wmin.y);
        float scale = arrowLength * worldSize;
        Vector3 a = new Vector3(selPoint.x, selPoint.y, 0f);
        Vector3 b = new Vector3(selPoint.x + dir.x * scale, selPoint.y + dir.y * scale, 0f);
        gradArrow.widthMultiplier = lineWidth;
        gradArrow.positionCount = 2; gradArrow.SetPosition(0, a); gradArrow.SetPosition(1, b);
    }

    void DrawEpsRing()
    {
        if (!epsRing) return;
        if (!(tglShowRing && tglShowRing.isOn) || selIndex < 0) { epsRing.positionCount = 0; return; }
        float eps = sldEps ? sldEps.value : 0.15f;
        int norm = drpNorm ? drpNorm.value : 0; // 0 L2, 1 Linf, 2 L1
        const int N = 64;
        if (norm == 0)
        { // circle (L2)
            epsRing.positionCount = N + 1;
            for (int i = 0; i <= N; i++)
            {
                float th = (2 * Mathf.PI * i) / N;
                Vector3 p = new Vector3(selPoint.x + eps * Mathf.Cos(th), selPoint.y + eps * Mathf.Sin(th), 0f);
                epsRing.SetPosition(i, p);
            }
        }
        else if (norm == 1)
        { // square (Linf)
            Vector3[] pts = SquareLoop(selPoint, eps); epsRing.positionCount = pts.Length; epsRing.SetPositions(pts);
        }
        else
        { // diamond (L1)
            Vector3[] pts = DiamondLoop(selPoint, eps); epsRing.positionCount = pts.Length; epsRing.SetPositions(pts);
        }
    }

    Vector3[] SquareLoop(Vector2 c, float r)
    {
        // closed loop clockwise
        return new Vector3[]{
            new Vector3(c.x-r,c.y-r,0), new Vector3(c.x+r,c.y-r,0),
            new Vector3(c.x+r,c.y+r,0), new Vector3(c.x-r,c.y+r,0),
            new Vector3(c.x-r,c.y-r,0)
        };
    }
    Vector3[] DiamondLoop(Vector2 c, float r)
    {
        return new Vector3[]{
            new Vector3(c.x,  c.y-r,0), new Vector3(c.x+r,c.y,0),
            new Vector3(c.x,  c.y+r,0), new Vector3(c.x-r,c.y,0),
            new Vector3(c.x,  c.y-r,0)
        };
    }

    void DrawPath(List<Vector3> pts)
    {
        if (!pathLine) return;
        if (!(tglShowPath && tglShowPath.isOn)) { pathLine.positionCount = 0; return; }
        pathLine.positionCount = pts.Count;
        for (int i = 0; i < pts.Count; i++) pathLine.SetPosition(i, pts[i]);
    }

    void DrawGhost(Vector2 x)
    {
        if (!advGhost) return;
        advGhost.enabled = true;
        advGhost.transform.position = new Vector3(x.x, x.y, 0f);
    }

    // ---------- Utils ----------
    float Acc(float[,] P, float[,] Y, float thr) { int N = P.GetLength(0), ok = 0; for (int i = 0; i < N; i++) { int h = P[i, 0] >= thr ? 1 : 0; int y = Y[i, 0] > 0.5f ? 1 : 0; if (h == y) ok++; } return ok / (float)Mathf.Max(1, N); }
    int[] SampleBatch(int bs, int total) { var set = new HashSet<int>(); while (set.Count < bs) set.Add(rnd.Next(total)); var arr = new int[bs]; int k = 0; foreach (var i in set) arr[k++] = i; return arr; }
    Tuple<float[,], float[,]> Gather(int[] idx, float[,] X, float[,] Y) { var Xb = new float[idx.Length, 2]; var Yb = new float[idx.Length, 1]; for (int r = 0; r < idx.Length; r++) { int i = idx[r]; Xb[r, 0] = X[i, 0]; Xb[r, 1] = X[i, 1]; Yb[r, 0] = Y[i, 0]; } return new Tuple<float[,], float[,]>(Xb, Yb); }

    void OnValidate()
    {
        // live updates for style when tweaking in inspector
        for (int i = 0; i < dots.Count; i++)
        {
            dots[i].transform.localScale = Vector3.one * ((i == selIndex) ? selectedDotScale : dotScale);
            if (i < dataset?.labels?.Length) dots[i].color = dataset.labels[i] > 0.5f ? colPos : colNeg;
        }
        ApplyStyle();
        RedrawAttribution();
    }
}
