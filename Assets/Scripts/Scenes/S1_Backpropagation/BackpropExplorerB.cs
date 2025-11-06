using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System;
using System.Collections.Generic;

public class BackpropExplorerB : MonoBehaviour
{
    [Header("Refs")]
    public Dataset2D dataset;                 // assign asset; we clone it at runtime
    public DecisionFieldRenderer field;
    public Transform pointsParent;
    public GameObject pointPrefab;

    [Header("UI")]
    public Button btnStep;
    public Toggle tglAuto;
    public Slider sldLR;
    public TMP_Dropdown drpActivation;        // 0=Tanh, 1=ReLU, 2=Sigmoid
    public TMP_Text txtLoss;

    [Header("Minibatch UI")]
    public Toggle tglMinibatch;
    public Slider sldBatch;
    public TMP_Text txtBatch;

    [Header("Extra UI")]
    public Button btnReset;                   // Reset model weights + clear chart
    public Button btnShuffle;                 // Regenerate dataset (new seed/pose)

    [Header("Optional visuals")]
    public LossChartB lossChart;              // rolling loss graph (RawImage)
    public GradientEdgeOverlayB overlay;      // |∂L/∂w| thickness, |∂L/∂b| halos
    public ActivationPanelB activationPanel;  // node activations + weight/grad labels

    [Header("Optional indicators")]
    public AccuracyBadgeB accuracyBadge;      // shows "Accuracy: xx% Step: n"
    public ProbabilityProbeB probe;           // mouse probe (set its mlp)

    [Header("Optional: highlight misclassified points")]
    public bool highlightMistakes = false;
    public Color wrongTint = new Color(1f, 0.95f, 0.3f, 1f); // yellowish

    // --- internals ---
    MLP mlp;
    float[,] X, Y;

    bool autoTrain = false;
    float autoTimer = 0f;
    const float autoInterval = 0.05f;

    readonly List<GameObject> pointGOs = new List<GameObject>();
    Color[] basePointColors = Array.Empty<Color>();

    int[] lastBatchIdx = Array.Empty<int>();
    float highlightTimer = 0f;
    const float highlightDuration = 0.20f;
    System.Random rnd = new System.Random(1234);

    int stepCount = 0;
    float prevLoss = 0f;

    void Start()
    {
        // Make a runtime copy and randomize every play
        if (dataset == null)
        {
            dataset = ScriptableObject.CreateInstance<Dataset2D>();
            dataset.GenerateBlobs();
        }
        else
        {
            dataset = ScriptableObject.Instantiate(dataset); // don't mutate the asset
            dataset.ReseedAndGenerateClean();                // rotated/translated clean blobs
        }

        // Init model: 2→3→1, linear output + BCE, hidden activation from dropdown
        mlp = new MLP(2, 3, 1, seed: UnityEngine.Random.Range(1, 1_000_000)) { lossType = LossType.BCE };
        mlp.activation = (Act)drpActivation.value;

        // UI hooks
        btnStep.onClick.AddListener(StepTrain);
        tglAuto.onValueChanged.AddListener(v => autoTrain = v);
        sldLR.onValueChanged.AddListener(v => mlp.lr = v);
        mlp.lr = sldLR.value;

        drpActivation.onValueChanged.AddListener(OnActChanged);
        if (tglMinibatch) tglMinibatch.onValueChanged.AddListener(_ => UpdateBatchLabel());
        if (sldBatch) sldBatch.onValueChanged.AddListener(_ => UpdateBatchLabel());
        UpdateBatchLabel();

        if (btnReset) btnReset.onClick.AddListener(ResetModel);
        if (btnShuffle) btnShuffle.onClick.AddListener(ShuffleData);

        X = dataset.XMatrix();
        Y = dataset.YMatrix();

        SpawnPoints();

        // prime UI
        RedrawField();
        var (loss, _) = mlp.Forward(X, Y);
        prevLoss = loss;
        UpdateLossText();
        PushLossToChart();
        NotifyPanels();
        UpdateAccuracyBadge();

        // optional probe
        if (probe != null) probe.mlp = mlp;

        // initial faithfulness report
        ReportFaithfulnessToTracker();
    }

    void Update()
    {
        if (autoTrain)
        {
            autoTimer += Time.deltaTime;
            if (autoTimer >= autoInterval)
            {
                autoTimer = 0f;
                StepTrain();
            }
        }

        // shrink highlighted minibatch points back to normal
        if (highlightTimer > 0f)
        {
            highlightTimer -= Time.deltaTime;
            float t = Mathf.InverseLerp(0f, highlightDuration, highlightTimer);
            float scale = Mathf.Lerp(0.15f, 0.20f, t);
            foreach (var idx in lastBatchIdx)
                if ((uint)idx < (uint)pointGOs.Count)
                    pointGOs[idx].transform.localScale = Vector3.one * scale;
        }

        // shortcuts
        if (Input.GetKeyDown(KeyCode.Space)) StepTrain();
        if (Input.GetKeyDown(KeyCode.A)) tglAuto.isOn = !tglAuto.isOn;
        if (Input.GetKeyDown(KeyCode.R)) ResetModel();
        if (Input.GetKeyDown(KeyCode.S)) ShuffleData();
    }

    void OnActChanged(int idx)
    {
        mlp.activation = (Act)idx; // hidden-layer activation
        RedrawField();
        NotifyPanels();
        UpdateAccuracyBadge();
        if (highlightMistakes) MarkMistakes();

        // Gamification: record tried variant + re-check faithfulness
        var tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();
        tracker?.ReportTriedVariant(((Act)idx).ToString());
        ReportFaithfulnessToTracker();
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
            sr.color = dataset.labels[i] > 0.5f ? new Color(0.94f, 0.28f, 0.28f) : new Color(0.28f, 0.46f, 0.94f);
            sr.sortingOrder = 10;
            go.transform.localScale = Vector3.one * 0.15f;
            pointGOs.Add(go);
        }

        // cache base colors for mistake-highlighting toggles
        basePointColors = new Color[pointGOs.Count];
        for (int i = 0; i < pointGOs.Count; i++)
            basePointColors[i] = pointGOs[i].GetComponent<SpriteRenderer>().color;
    }

    void StepTrain()
    {
        float loss;

        if (tglMinibatch && tglMinibatch.isOn)
        {
            int bs = Mathf.Clamp((int)sldBatch.value, 1, dataset.count);
            lastBatchIdx = SampleBatch(bs, dataset.count);
            var (Xb, Yb) = GatherBatch(lastBatchIdx);

            (loss, _) = mlp.Forward(Xb, Yb);
            mlp.StepSGD(bs);

            // pop current batch
            highlightTimer = highlightDuration;
            foreach (var idx in lastBatchIdx)
                if ((uint)idx < (uint)pointGOs.Count)
                    pointGOs[idx].transform.localScale = Vector3.one * 0.22f;
        }
        else
        {
            (loss, _) = mlp.Forward(X, Y);
            mlp.StepSGD(dataset.count);
            lastBatchIdx = Array.Empty<int>();
        }

        txtLoss.text = $"Loss: {loss:F4} | LR: {mlp.lr:F4} | Act: {mlp.activation}";
        RedrawField();
        PushLossToChart();
        NotifyPanels();

        stepCount++;
        UpdateAccuracyBadge();

        if (highlightMistakes) MarkMistakes();

        prevLoss = loss;

        // Gamification: count action + report faithfulness
        var tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();
        tracker?.ReportAction("weight_adjust");
        ReportFaithfulnessToTracker();
    }

    void RedrawField() => field.Redraw(WorldToProb);

    float WorldToProb(Vector2 w)
    {
        float[,] Xi = new float[1, 2] { { w.x, w.y } };
        float[,] Yi = new float[1, 1] { { 0f } };
        var (_, pred) = mlp.Forward(Xi, Yi);
        return pred[0, 0];
    }

    int[] SampleBatch(int bs, int n)
    {
        int[] idx = new int[bs];
        for (int i = 0; i < bs; i++) idx[i] = rnd.Next(n);
        return idx;
    }

    (float[,], float[,]) GatherBatch(int[] idx)
    {
        int bs = idx.Length;
        float[,] Xb = new float[bs, 2];
        float[,] Yb = new float[bs, 1];
        for (int i = 0; i < bs; i++)
        {
            Xb[i, 0] = X[idx[i], 0];
            Xb[i, 1] = X[idx[i], 1];
            Yb[i, 0] = Y[idx[i], 0];
        }
        return (Xb, Yb);
    }

    void UpdateBatchLabel()
    {
        if (txtBatch == null) return;
        int bs = (int)sldBatch.value;
        txtBatch.text = $"Batch: {bs}";
    }

    void PushLossToChart()
    {
        var (loss, _) = mlp.Forward(X, Y);
        lossChart.Push(loss);
    }

    void NotifyPanels()
    {
        overlay?.SyncFromMLP(mlp);
        activationPanel?.Render(mlp, dataset);
    }

    void UpdateAccuracyBadge()
    {
        accuracyBadge?.UpdateFrom(mlp, X, Y, stepCount);
    }

    // NEW: re-usable loss text (was missing and caused your error)
    void UpdateLossText()
    {
        if (mlp == null || X == null || Y == null || txtLoss == null) return;
        var (loss, _) = mlp.Forward(X, Y);
        txtLoss.text = $"Loss: {loss:F4} | LR: {mlp.lr:F4} | Act: {mlp.activation}";
    }

    // --- extra actions ---
    void ResetModel()
    {
        mlp = new MLP(2, 3, 1, seed: UnityEngine.Random.Range(1, 1_000_000))
        {
            lossType = LossType.BCE,
            activation = (Act)drpActivation.value,
            lr = sldLR.value
        };
        stepCount = 0;
        lossChart?.ClearSeries();
        UpdateLossText();
        RedrawField();
        NotifyPanels();
        UpdateAccuracyBadge();

        if (probe != null) probe.mlp = mlp;
        if (highlightMistakes) MarkMistakes();

        // Recompute/Report F after reset
        ReportFaithfulnessToTracker();
    }

    void ShuffleData()
    {
        dataset.ReseedAndGenerateClean();
        X = dataset.XMatrix();
        Y = dataset.YMatrix();
        SpawnPoints();
        stepCount = 0;

        UpdateLossText();
        RedrawField();
        NotifyPanels();
        UpdateAccuracyBadge();
        // keep chart history; uncomment to clear:
        // lossChart?.ClearSeries();

        if (highlightMistakes) MarkMistakes();

        // Recompute/Report F after shuffle
        ReportFaithfulnessToTracker();
    }

    // --- optional: misclassification tinting ---
    void MarkMistakes()
    {
        var (_, P) = mlp.Forward(X, Y);
        for (int i = 0; i < dataset.count; i++)
        {
            bool wrong = (P[i, 0] >= 0.5f) != (dataset.labels[i] > 0.5f);
            var sr = pointGOs[i].GetComponent<SpriteRenderer>();
            var baseCol = basePointColors[i];
            sr.color = wrong ? Color.Lerp(baseCol, wrongTint, 0.55f) : baseCol;
        }
    }

    // --- helper: compute current accuracy (used as faithfulness proxy) ---
    float ComputeFaithfulness()
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
        return n > 0 ? (float)correct / n : 0f; // 0..1
    }

    void ReportFaithfulnessToTracker()
    {
        var tracker = UnityEngine.Object.FindFirstObjectByType<ObjectiveTracker>();
        if (tracker == null) return;
        tracker.ReportFaithfulness(ComputeFaithfulness());
    }
}
