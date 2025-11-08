using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System;
using System.Collections.Generic;

public class LossThresholdExplorer : MonoBehaviour
{
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
    MLP mlp;
    float[,] X, Y;
    readonly List<GameObject> pointGOs = new();
    System.Random rnd = new System.Random(7);
    bool auto = false;
    float timer = 0f;
    const float dt = 0.05f;

    // Gamification
    ObjectiveTracker _tracker;

    void Start()
    {
        dataset = dataset ? ScriptableObject.Instantiate(dataset) : ScriptableObject.CreateInstance<Dataset2D>();
        if (dataset.points == null || dataset.points.Length == 0)
            dataset.GenerateBlobs();
        dataset.ReseedAndGenerateClean();
        X = dataset.XMatrix();
        Y = dataset.YMatrix();

        mlp = new MLP(2, 3, 1, seed: UnityEngine.Random.Range(1, 1_000_000))
        {
            lossType = LossType.BCE,
            lr = sldLR.value
        };

        SpawnPoints();

        // cache tracker (if present)
        _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();

        // UI hooks
        btnStep.onClick.AddListener(Step);
        tglAuto.onValueChanged.AddListener(v => auto = v);
        tglMinibatch.onValueChanged.AddListener(_ => { });
        sldBatch.onValueChanged.AddListener(_ => { });
        sldLR.onValueChanged.AddListener(v => mlp.lr = v);
        drpLossType.onValueChanged.AddListener(OnLossChanged);
        // threshold slider: refresh + gamification event
        sldThreshold.onValueChanged.AddListener(OnThresholdSliderChanged);
        btnReset.onClick.AddListener(ResetAll);
        btnShuffle.onClick.AddListener(ShuffleData);

        RefreshAll();               // computes metrics & draws
        // initial metrics -> tracker
        ReportMetricsToTrackerFromCurrent();
    }

    void Update()
    {
        if (!auto) return;
        timer += Time.deltaTime;
        if (timer >= dt)
        {
            timer = 0f;
            Step();
        }
    }

    void OnLossChanged(int idx)
    {
        mlp.lossType = idx == 0 ? LossType.BCE : LossType.MSE;
        RefreshAll();
        ReportMetricsToTrackerFromCurrent();
    }

    void OnThresholdSliderChanged(float _)
    {
        RefreshPanelsOnly();
        ReportThresholdChange();
        ReportMetricsToTrackerFromCurrent();
    }

    void Step()
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
        RefreshAll();
        ReportMetricsToTrackerFromCurrent();
    }

    void RefreshAll()
    {
        var (loss, P) = mlp.Forward(X, Y);

        float thr = sldThreshold.value;
        (int TP, int FP, int TN, int FN, float prec, float rec, float f1, float acc) = Metrics(P, Y, thr);

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

    void RefreshPanelsOnly()
    {
        var (loss, P) = mlp.Forward(X, Y);
        float thr = sldThreshold.value;

        (int TP, int FP, int TN, int FN, float prec, float rec, float f1, float acc) = Metrics(P, Y, thr);

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

    void ResetAll()
    {
        mlp = new MLP(2, 3, 1, seed: UnityEngine.Random.Range(1, 1_000_000))
        {
            lossType = (drpLossType.value == 0 ? LossType.BCE : LossType.MSE),
            lr = sldLR.value
        };
        RefreshAll();
        ReportMetricsToTrackerFromCurrent();
    }

    void ShuffleData()
    {
        dataset.ReseedAndGenerateClean();
        X = dataset.XMatrix();
        Y = dataset.YMatrix();
        SpawnPoints();
        RefreshAll();
        ReportMetricsToTrackerFromCurrent();
    }

    void SpawnPoints()
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

    void UpdatePointStyles(float[,] P)
    {
        for (int i = 0; i < pointGOs.Count; i++)
        {
            var sr = pointGOs[i].GetComponent<SpriteRenderer>();
            bool isRed = dataset.labels[i] > 0.5f;
            Color baseCol = isRed ? redColor : blueColor;

            // confidence-based fade (uncertain points are lighter)
            float conf = Mathf.Clamp01(Mathf.Abs(P[i, 0] - 0.5f) * 2f);
            float a = Mathf.Lerp(0.25f, baseCol.a, conf);
            baseCol.a = a;
            sr.color = baseCol;
        }
    }

    // metrics
    (int, int, int, int, float, float, float, float) Metrics(float[,] P, float[,] Y, float thr)
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

    // ---------- Gamification glue ----------

    void ReportThresholdChange()
    {
        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();
        _tracker?.ReportAction("threshold_change");
    }

    void ReportMetricsToTrackerFromCurrent()
    {
        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();
        if (_tracker == null || mlp == null || X == null || Y == null) return;

        var (loss, P) = mlp.Forward(X, Y);
        float thr = sldThreshold.value;
        (_, _, _, _, _, _, _, float acc) = Metrics(P, Y, thr);

        _tracker.ReportAccuracy(Mathf.Clamp01(acc));

        float F = ComputeFaithfulness();
        _tracker.ReportFaithfulness(F);
    }

    // Perturbation-based faithfulness: stability of correct predictions
    float ComputeFaithfulness()
    {
        if (mlp == null || X == null || Y == null) return 0f;
        int n = X.GetLength(0);
        if (n == 0) return 0f;

        // bounds
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
            if (pred != lab) continue; // only evaluate regions we currently classify correctly

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

    float NextGaussian()
    {
        double u1 = 1.0 - rnd.NextDouble();
        double u2 = 1.0 - rnd.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return (float)randStdNormal;
    }

    // Call this from your "Done / Next" button for S4_UNDER240
    public void OnSceneCompleted()
    {
        if (_tracker == null)
            _tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();
        _tracker?.ReportSceneFinish();
    }
}
