using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System;
using System.Collections.Generic;

public class ActivationExplorer : MonoBehaviour
{
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
    public Color redColor = new Color(0.94f, 0.45f, 0.45f, 1f); // calmer red

    // --- internals ---
    MLP mlp;
    float[,] X, Y;
    readonly List<GameObject> pointGOs = new List<GameObject>();
    System.Random rnd = new System.Random(42);

    bool auto = false;
    float timer = 0f;
    const float autoDt = 0.05f;

    void Start()
    {
        // Dataset: runtime clone + randomized pose for each run
        dataset = dataset ? ScriptableObject.Instantiate(dataset)
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
            lr = sldLR.value
        };

        // UI hooks
        if (btnStep) btnStep.onClick.AddListener(Step);
        if (tglAuto) tglAuto.onValueChanged.AddListener(v => auto = v);
        if (tglMinibatch) tglMinibatch.onValueChanged.AddListener(_ => { });
        if (sldBatch) sldBatch.onValueChanged.AddListener(_ => { });
        if (sldLR) sldLR.onValueChanged.AddListener(v => mlp.lr = v);
        if (drpActivation) drpActivation.onValueChanged.AddListener(OnActivationChanged);
        if (btnReset) btnReset.onClick.AddListener(ResetModel);
        if (btnShuffle) btnShuffle.onClick.AddListener(ShuffleData);
        if (tglShowBoundary) tglShowBoundary.onValueChanged.AddListener(_ => Redraw());

        // Beginner mode toggle
        if (tglBeginner) tglBeginner.onValueChanged.AddListener(v => {
            if (explainer) explainer.gameObject.SetActive(v);
        });

        // Points
        SpawnPoints();

        // Prime optional mini-plot & explainer visibility
        actCurve?.SetAct((Act)drpActivation.value);
        if (explainer) explainer.gameObject.SetActive(tglBeginner ? tglBeginner.isOn : true);

        // Initial draw & styles
        Redraw();
        UpdateLossText();
        UpdatePointStyles();
    }

    void Update()
    {
        if (auto)
        {
            timer += Time.deltaTime;
            if (timer >= autoDt) { timer = 0f; Step(); }
        }

        // Handy shortcuts
        if (Input.GetKeyDown(KeyCode.Space)) Step();
        if (Input.GetKeyDown(KeyCode.A)) if (tglAuto) tglAuto.isOn = !tglAuto.isOn;
        if (Input.GetKeyDown(KeyCode.R)) ResetModel();
        if (Input.GetKeyDown(KeyCode.S)) ShuffleData();
        if (Input.GetKeyDown(KeyCode.B)) if (tglShowBoundary) tglShowBoundary.isOn = !tglShowBoundary.isOn;
        if (Input.GetKeyDown(KeyCode.H)) if (tglBeginner) tglBeginner.isOn = !tglBeginner.isOn;
    }

    void OnActivationChanged(int idx)
    {
        mlp.activation = (Act)idx;
        actCurve?.SetAct((Act)idx);
        Redraw();
        UpdateLossText();
        UpdatePointStyles();
    }

    void Step()
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

        if (txtLoss) txtLoss.text = $"Loss: {loss:F4} | LR: {mlp.lr:F4} | Act: {mlp.activation}";
        Redraw();
        UpdatePointStyles();   // refresh alpha/desaturation with new predictions
    }

    void Redraw()
    {
        // Per-neuron geometry
        hinges?.Render(mlp, dataset);

        // 0.5 decision curve (prefer crisp GL curve; fall back to sampler if provided)
        bool showBoundary = tglShowBoundary ? tglShowBoundary.isOn : true;

        if (showBoundary && boundaryCurve)
            boundaryCurve.Redraw(WorldToProb);

        if (boundarySampler)
        {
            boundarySampler.enabled = showBoundary && !boundaryCurve;
            if (boundarySampler.enabled)
                boundarySampler.Redraw(WorldToProb);
        }

        // Stats + Explainer panels
        actStats?.UpdateFrom(mlp, dataset);
        explainer?.UpdateFrom(mlp, dataset);
    }

    float WorldToProb(Vector2 w)
    {
        float[,] Xi = new float[1, 2] { { w.x, w.y } };
        float[,] Yi = new float[1, 1] { { 0f } };
        var (_, pred) = mlp.Forward(Xi, Yi);
        return Mathf.Clamp01(pred[0, 0]);
    }

    void UpdateLossText()
    {
        var (loss, _) = mlp.Forward(X, Y);
        if (txtLoss) txtLoss.text = $"Loss: {loss:F4} | LR: {mlp.lr:F4} | Act: {mlp.activation}";
    }

    void ResetModel()
    {
        mlp = new MLP(2, 3, 1, seed: UnityEngine.Random.Range(1, 1_000_000))
        {
            lossType = LossType.BCE,
            activation = (Act)drpActivation.value,
            lr = sldLR ? sldLR.value : mlp.lr
        };
        actCurve?.SetAct((Act)drpActivation.value);
        Redraw();
        UpdateLossText();
        UpdatePointStyles();
    }

    void ShuffleData()
    {
        dataset.ReseedAndGenerateClean();
        X = dataset.XMatrix(); Y = dataset.YMatrix();
        SpawnPoints();
        Redraw();
        UpdateLossText();
        UpdatePointStyles();
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
            // initial soft color (will be refined by UpdatePointStyles)
            sr.color = dataset.labels[i] > 0.5f ? redColor : blueColor;
            sr.sortingOrder = 10;
            go.transform.localScale = Vector3.one * dotScale;
            pointGOs.Add(go);
        }
    }

    // --- style refresh for dots ---
    void UpdatePointStyles()
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

            // confidence (0..1): 0 uncertain, 1 very sure
            float conf = 1f;
            if (P != null) conf = Mathf.Clamp01(Mathf.Abs(P[i, 0] - 0.5f) * 2f);

            // alpha
            float a = fadeByConfidence ? Mathf.Lerp(alphaMin, alphaMax, conf) : alphaMax;

            // desaturate towards gray when uncertain
            if (desaturateUncertain)
            {
                float g = baseCol.grayscale;
                Color gray = new Color(g, g, g, 1f);
                float t = desaturateStrength * (1f - conf); // more desat when uncertain
                baseCol = Color.Lerp(baseCol, gray, t);
            }

            baseCol.a = a;
            sr.color = baseCol;

            // size
            pointGOs[i].transform.localScale = Vector3.one * dotScale;
        }
    }

    // --- minibatch helpers ---
    int[] SampleBatch(int bs, int total)
    {
        var set = new HashSet<int>();
        while (set.Count < bs) set.Add(rnd.Next(total));
        var arr = new int[bs]; int k = 0; foreach (var i in set) arr[k++] = i;
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
}
