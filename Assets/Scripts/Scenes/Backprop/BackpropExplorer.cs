using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class BackpropExplorer : MonoBehaviour
{
    [Header("Refs")]
    public Dataset2D dataset;
    public DecisionFieldRenderer field;
    public Transform pointsParent;
    public GameObject pointPrefab; // small circle sprite (white), we’ll tint by label

    [Header("UI")]
    public Button btnStep;
    public Toggle tglAuto;
    public Slider sldLR;
    public TMP_Dropdown drpActivation;
    public TMP_Text txtLoss;

    MLP mlp;
    float[,] X, Y;
    bool autoTrain = false;
    float autoTimer = 0f;
    const float autoInterval = 0.05f;

    void Start()
    {
        if (dataset == null)
        {
            dataset = ScriptableObject.CreateInstance<Dataset2D>();
            dataset.GenerateBlobs();
        }
        else if (dataset.points == null || dataset.points.Length == 0)
        {
            dataset.GenerateBlobs();
        }

        // Model: 2 -> 3 -> 1
        mlp = new MLP(2, 3, 1, seed: 123);
        mlp.lossType = LossType.BCE;

        // Hook UI
        btnStep.onClick.AddListener(StepTrain);
        tglAuto.onValueChanged.AddListener(v => autoTrain = v);
        sldLR.onValueChanged.AddListener(v => mlp.lr = v);
        mlp.lr = sldLR.value;

        drpActivation.onValueChanged.AddListener(OnActChanged);
        OnActChanged(drpActivation.value);

        // Prepare data
        X = dataset.XMatrix();
        Y = dataset.YMatrix();

        // Spawn points
        SpawnPoints();

        // First draw
        RedrawField();
        UpdateLossText();
    }

    void OnActChanged(int idx)
    {
        mlp.activation = (Act)idx; // order matches dropdown: 0=Tanh, 1=ReLU, 2=Sigmoid
        RedrawField();
    }

    void SpawnPoints()
    {
        // Clear previous
        for (int i = pointsParent.childCount - 1; i >= 0; i--) Destroy(pointsParent.GetChild(i).gameObject);

        for (int i = 0; i < dataset.points.Length; i++)
        {
            var go = Instantiate(pointPrefab, dataset.points[i], Quaternion.identity, pointsParent);
            var sr = go.GetComponent<SpriteRenderer>();
            sr.color = dataset.labels[i] > 0.5f ? new Color(0.9f, 0.2f, 0.2f) : new Color(0.2f, 0.4f, 0.9f);
            sr.sortingOrder = 10;
            go.transform.localScale = Vector3.one * 0.15f;
        }
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
    }

    void StepTrain()
    {
        var (loss, _) = mlp.Forward(X, Y);
        mlp.StepSGD(dataset.count);
        UpdateLossText();
        RedrawField();
    }

    void UpdateLossText()
    {
        var (loss, _) = mlp.Forward(X, Y);
        txtLoss.text = $"Loss: {loss:F4} | LR: {mlp.lr:F4} | Act: {mlp.activation}";
    }

    void RedrawField()
    {
        field.Redraw(WorldToProb);
    }

    float Prob(Vector2 x)
    {
        // Single inference for one sample
        float[,] Xi = new float[1, 2] { { x.x, x.y } };
        float[,] Yi = new float[1, 1] { { 0f } };
        var (_, pred) = mlp.Forward(Xi, Yi);
        return pred[0, 0];
    }

    float WorldToProb(Vector2 w) => Mathf.Clamp01(Prob(w));
}
