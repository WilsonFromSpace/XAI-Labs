using UnityEngine;
using System.Collections.Generic;
using TMPro;

/// <summary>
/// Draws per-neuron hinge lines and sensitivity bands directly in world space.
/// Thickness/alpha ~ mean |dL/dz| per hidden neuron. No probability heatmap involved.
/// </summary>
[RequireComponent(typeof(Transform))]
public class ActivationHingesOverlay : MonoBehaviour
{
    [Header("World bounds (same units as your points)")]
    public Vector2 worldMin = new Vector2(-5, -5);
    public Vector2 worldMax = new Vector2(5, 5);

    [Header("Style")]
    public Material lineMat;
    public Gradient colorPerNeuron; // use 3 distinct colors for the 3 hidden units
    public float baseWidth = 0.06f;
    public float bandOffsetTanh = 1.0f;   // z = ±k
    public float bandOffsetSigmoid = 2.0f;
    public float bandOffsetReLU = 0.4f;   // show one-sided band at z=+k

    [Header("Labels (optional)")]
    public TMP_FontAsset font;
    public float fontSize = 2.8f;

    readonly List<LineRenderer> center = new();
    readonly List<LineRenderer> bandPos = new();
    readonly List<LineRenderer> bandNeg = new();
    readonly List<TextMeshPro> labels = new();

    public void ClearAll()
    {
        foreach (var lr in center) if (lr) Destroy(lr.gameObject);
        foreach (var lr in bandPos) if (lr) Destroy(lr.gameObject);
        foreach (var lr in bandNeg) if (lr) Destroy(lr.gameObject);
        foreach (var t in labels) if (t) Destroy(t.gameObject);
        center.Clear(); bandPos.Clear(); bandNeg.Clear(); labels.Clear();
    }

    LineRenderer NewLR(string name, Color c)
    {
        var go = new GameObject(name);
        go.transform.SetParent(transform, false);
        var lr = go.AddComponent<LineRenderer>();
        lr.material = lineMat;
        lr.positionCount = 2;
        lr.numCapVertices = 8;
        lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        lr.receiveShadows = false;
        lr.useWorldSpace = true;
        lr.sortingOrder = 200;
        lr.startColor = lr.endColor = c;
        lr.startWidth = lr.endWidth = baseWidth;
        return lr;
    }

    TextMeshPro NewLabel(string name, Color c)
    {
        var go = new GameObject(name);
        go.transform.SetParent(transform, false);
        var tmp = go.AddComponent<TextMeshPro>();
        tmp.font = font;
        tmp.fontSize = fontSize;
        tmp.alignment = TextAlignmentOptions.Center;
        tmp.color = c;
        tmp.text = "";
        tmp.enableVertexGradient = false;
        tmp.raycastTarget = false;
        tmp.GetComponent<RectTransform>().sizeDelta = new Vector2(8, 2);
        return tmp;
    }

    // Main entry point: call after each train step / reset / shuffle
    public void Render(MLP mlp, Dataset2D data)
    {
        if (mlp == null || data == null) return;

        // ensure 3 tracks (for 3 hidden units)
        int H = mlp.Ls[0].b.Length;
        while (center.Count < H) center.Add(NewLR("Hinge", Color.white));
        while (bandPos.Count < H) bandPos.Add(NewLR("Band+", Color.white));
        while (bandNeg.Count < H) bandNeg.Add(NewLR("Band-", Color.white));
        while (labels.Count < H) labels.Add(NewLabel("Lbl", Color.white));

        // compute mean |dL/dz| for each hidden neuron using the full dataset
        float[] sens = MeanAbs_dLdz(mlp, data);

        for (int j = 0; j < H; j++)
        {
            // color for neuron j
            Color cj = colorPerNeuron.Evaluate(H == 1 ? 0f : j / (float)(H - 1));

            // pull weights and bias of hidden neuron j
            float w0 = mlp.Ls[0].W[0, j];
            float w1 = mlp.Ls[0].W[1, j];
            float b = mlp.Ls[0].b[j];
            var w = new Vector2(w0, w1);
            float wnorm = Mathf.Max(1e-6f, w.magnitude);
            Vector2 n = w / wnorm;                 // unit normal to the hinge
            Vector2 t = new Vector2(-n.y, n.x);    // tangent direction along the hinge

            // choose band offset (in z-space → convert to distance by /|w|)
            float k = bandOffsetTanh;
            if (mlp.activation == Act.Sigmoid) k = bandOffsetSigmoid;
            if (mlp.activation == Act.ReLU) k = bandOffsetReLU;

            float d0 = 0f;        // center line (z = 0)
            float dPos = k / wnorm;
            float dNeg = -k / wnorm;

            // line segments within the world rect
            (Vector2 a0, Vector2 b0, bool ok0) = LineInRect(n, b, d0, worldMin, worldMax);
            (Vector2 aP, Vector2 bP, bool okP) = LineInRect(n, b, dPos, worldMin, worldMax);
            (Vector2 aN, Vector2 bN, bool okN) = LineInRect(n, b, dNeg, worldMin, worldMax);

            var lrC = center[j]; var lrP = bandPos[j]; var lrN = bandNeg[j];
            lrC.startColor = lrC.endColor = cj;
            lrP.startColor = lrP.endColor = new Color(cj.r, cj.g, cj.b, 0.65f);
            lrN.startColor = lrN.endColor = new Color(cj.r, cj.g, cj.b, mlp.activation == Act.ReLU ? 0.0f : 0.35f);

            float width = baseWidth * Mathf.Lerp(0.7f, 2.5f, Mathf.Clamp01(sens[j] / (0.2f + sens[j])));
            lrC.startWidth = lrC.endWidth = width;
            lrP.startWidth = lrP.endWidth = baseWidth * 0.9f;
            lrN.startWidth = lrN.endWidth = baseWidth * 0.9f;

            if (ok0) { lrC.enabled = true; lrC.SetPosition(0, a0); lrC.SetPosition(1, b0); } else lrC.enabled = false;
            if (okP) { lrP.enabled = true; lrP.SetPosition(0, aP); lrP.SetPosition(1, bP); } else lrP.enabled = false;
            if (okN) { lrN.enabled = true; lrN.SetPosition(0, aN); lrN.SetPosition(1, bN); } else lrN.enabled = false;

            // label near the center of the hinge
            var mid = (a0 + b0) * 0.5f;
            var lab = labels[j];
            lab.text = $"h{j}: |∂L/∂z|={sens[j]:0.000}";
            lab.transform.position = mid + t * 0.35f;
            lab.color = new Color(cj.r, cj.g, cj.b, 0.95f);
        }
    }

    // Compute mean absolute dL/dz for each hidden unit on the dataset
    float[] MeanAbs_dLdz(MLP mlp, Dataset2D data)
    {
        int N = data.count;
        int H = mlp.Ls[0].b.Length;
        float[] acc = new float[H];

        var X = data.XMatrix();
        var Y = data.YMatrix();

        // Forward to get caches + probabilities
        var (_, P) = mlp.Forward(X, Y);

        // BCE: dL/d(logit) = p - y  (since output is linear logits)
        var dZ1 = new float[N, 1];
        for (int i = 0; i < N; i++) dZ1[i, 0] = P[i, 0] - Y[i, 0];

        // dA0 = dZ1 * W1^T
        var W1T = TinyTensor.Transpose(mlp.Ls[1].W);
        var dA0 = TinyTensor.MatMul(dZ1, W1T);

        // dZ0 = dA0 ⊙ φ'(Z0)
        var (_, dphi) = Activations.Get(mlp.activation);
        var dphZ = TinyTensor.Apply(mlp.Ls[0].Z, dphi);         // using cached Z0
        var dZ0 = TinyTensor.Hadamard(dA0, dphZ);

        // mean |.| across samples
        for (int j = 0; j < H; j++)
        {
            float s = 0f;
            for (int i = 0; i < N; i++) s += Mathf.Abs(dZ0[i, j]);
            acc[j] = s / Mathf.Max(1, N);
        }
        return acc;
    }

    // Intersect a line n·x + b + d = 0 with axis-aligned rectangle [min,max]
    static (Vector2 A, Vector2 B, bool ok) LineInRect(Vector2 n, float b, float d, Vector2 mn, Vector2 mx)
    {
        // Parametric line: all points x with n·x + b + d = 0; direction t is perpendicular to n.
        // We intersect with the 4 rectangle edges and keep two farthest points within bounds.
        List<Vector2> pts = new List<Vector2>(4);
        // Edges: x=mn.x..mx.x at y=mn.y and y=mx.y; and y=mn.y..mx.y at x=mn.x and x=mx.x
        // Solve for x or y respectively.
        // y at x=mn.x
        if (Mathf.Abs(n.y) > 1e-6f)
        {
            float y = (-(b + d) - n.x * mn.x) / n.y;
            if (y >= mn.y - 1e-4f && y <= mx.y + 1e-4f) pts.Add(new Vector2(mn.x, y));
            y = (-(b + d) - n.x * mx.x) / n.y;
            if (y >= mn.y - 1e-4f && y <= mx.y + 1e-4f) pts.Add(new Vector2(mx.x, y));
        }
        if (Mathf.Abs(n.x) > 1e-6f)
        {
            float x = (-(b + d) - n.y * mn.y) / n.x;
            if (x >= mn.x - 1e-4f && x <= mx.x + 1e-4f) pts.Add(new Vector2(x, mn.y));
            x = (-(b + d) - n.y * mx.y) / n.x;
            if (x >= mn.x - 1e-4f && x <= mx.x + 1e-4f) pts.Add(new Vector2(x, mx.y));
        }
        if (pts.Count < 2) return (Vector2.zero, Vector2.zero, false);

        // choose the pair with maximal distance
        float best = -1f; Vector2 A = pts[0], B = pts[1];
        for (int i = 0; i < pts.Count; i++)
            for (int j = i + 1; j < pts.Count; j++)
            {
                float d2 = (pts[i] - pts[j]).sqrMagnitude;
                if (d2 > best) { best = d2; A = pts[i]; B = pts[j]; }
            }
        return (A, B, true);
    }
}
