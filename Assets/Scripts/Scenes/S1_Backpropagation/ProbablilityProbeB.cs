// Assets/Scripts/Scenes/Backprop/ProbabilityProbeB.cs
using UnityEngine;
using TMPro;

public class ProbabilityProbeB : MonoBehaviour
{
    public Camera cam; public MLP mlp; public TMP_Text label;
    public LineRenderer arrow; public float arrowLen = 0.8f;
    void Awake() { if (!cam) cam = Camera.main; if (arrow) { arrow.positionCount = 2; arrow.numCapVertices = 8; } }
    void Update()
    {
        if (mlp == null || cam == null) return;
        var w = cam.ScreenToWorldPoint(Input.mousePosition); w.z = 0f; transform.position = w;
        // forward to get hidden activations
        var L0 = mlp.Ls[0]; var L1 = mlp.Ls[1];
        float z0_0 = w.x * L0.W[0, 0] + w.y * L0.W[1, 0] + L0.b[0];
        float z0_1 = w.x * L0.W[0, 1] + w.y * L0.W[1, 1] + L0.b[1];
        float z0_2 = w.x * L0.W[0, 2] + w.y * L0.W[1, 2] + L0.b[2];
        var act = Activations.Get(mlp.activation);
        float a0_0 = act.f(z0_0), a0_1 = act.f(z0_1), a0_2 = act.f(z0_2);
        float z1 = a0_0 * L1.W[0, 0] + a0_1 * L1.W[1, 0] + a0_2 * L1.W[2, 0] + L1.b[0];
        float p = 1f / (1f + Mathf.Exp(-z1));
        if (label) label.text = $"p={p:0.000}";
        // gradient dp/dx (1x2): dp/dz1 = p(1-p); dZ1/dA0 = W1; dA0/dZ0 = φ'; dZ0/dX = W0^T
        float dp_dz1 = p * (1f - p);
        float g0 = dp_dz1 * L1.W[0, 0] * act.df(z0_0);
        float g1 = dp_dz1 * L1.W[1, 0] * act.df(z0_1);
        float g2 = dp_dz1 * L1.W[2, 0] * act.df(z0_2);
        Vector2 dp_dX = new Vector2(
          g0 * L0.W[0, 0] + g1 * L0.W[0, 1] + g2 * L0.W[0, 2],
          g0 * L0.W[1, 0] + g1 * L0.W[1, 1] + g2 * L0.W[1, 2]
        );
        if (arrow)
        {
            Vector2 a = (Vector2)w;
            Vector2 b = a + dp_dX.normalized * arrowLen * Mathf.Clamp(dp_dX.magnitude, 0.1f, 1f);
            arrow.SetPosition(0, a); arrow.SetPosition(1, b);
        }
    }
}
