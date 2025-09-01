using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;

public class ActivationPanelB : MonoBehaviour
{
    [Header("UI")]
    public RawImage img;

    [Header("Colors")]
    public Color bg = new Color(0.07f, 0.07f, 0.07f, 1);
    public Color ring = new Color(1, 1, 1, 0.9f);
    public Color edge = new Color(1, 1, 1, 0.28f);
    public Color fillBase = new Color(1f, 0.85f, 0.55f, 1f);
    public Color gradPos = new Color(0.95f, 0.35f, 0.35f, 1f);  // +gradient
    public Color gradNeg = new Color(0.35f, 0.55f, 0.95f, 1f);  // -gradient

    [Header("Text labels")]
    public bool showWeights = true;
    public bool showBiases = true;
    public float fontSize = 14f;

    Texture2D tex;
    const int W = 420, H = 220;

    // Layout (pixel coords inside the texture)
    Vector2[] posIn = { new Vector2(40, 160), new Vector2(40, 60) };
    Vector2[] posH = { new Vector2(210, 180), new Vector2(210, 110), new Vector2(210, 40) };
    Vector2 posO = new Vector2(360, 110);

    // Runtime-created TMP labels
    readonly List<TextMeshProUGUI> labels = new List<TextMeshProUGUI>();
    RectTransform rt; // this panel's rect

    void Awake()
    {
        if (img == null) img = GetComponent<RawImage>();
        rt = GetComponent<RectTransform>();

        tex = new Texture2D(W, H, TextureFormat.RGBA32, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        img.texture = tex;

        Clear();
        tex.Apply(false);
    }

    // Call after each train step / reset / shuffle
    public void Render(MLP mlp, Dataset2D data)
    {
        if (mlp == null || data == null) return;
        Clear();

        // ---- ACTIVATION PREVIEW (dataset-mean) ----
        float[,] X = data.XMatrix();
        int n = X.GetLength(0);
        int h = mlp.Ls[0].b.Length;

        var meanH = new float[h];
        float meanP = 0f;

        for (int i = 0; i < n; i++)
        {
            float[] a0 = ForwardHidden(mlp, X[i, 0], X[i, 1]);
            for (int j = 0; j < h; j++) meanH[j] += a0[j];
            meanP += ForwardProb(mlp, a0);
        }
        for (int j = 0; j < h; j++) meanH[j] /= n;
        meanP /= n;

        // normalize edge alpha by max |W|
        float maxW = 1e-6f;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < h; j++)
                maxW = Mathf.Max(maxW, Mathf.Abs(mlp.Ls[0].W[i, j]));
        for (int j = 0; j < h; j++)
            maxW = Mathf.Max(maxW, Mathf.Abs(mlp.Ls[1].W[j, 0]));

        // draw edges (input->hidden)
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < h; j++)
            {
                float a = Mathf.Abs(mlp.Ls[0].W[i, j]) / maxW;
                DrawLine(posIn[i], posH[j], new Color(edge.r, edge.g, edge.b, Mathf.Lerp(0.08f, 0.45f, a)));
            }
        // draw edges (hidden->output)
        for (int j = 0; j < h; j++)
        {
            float a = Mathf.Abs(mlp.Ls[1].W[j, 0]) / maxW;
            DrawLine(posH[j], posO, new Color(edge.r, edge.g, edge.b, Mathf.Lerp(0.08f, 0.45f, a)));
        }

        // nodes (rings + fills by activation)
        DrawNode(posIn[0], 16, 10, 0f);
        DrawNode(posIn[1], 16, 10, 0f);
        for (int j = 0; j < h; j++) DrawNode(posH[j], 18, 12, NormalizeAct(meanH[j], mlp.activation));
        DrawNode(posO, 20, 14, Mathf.Clamp01(meanP));

        // ---- TEXT LABELS (weights & biases with gradients) ----
        EnsureLabelPool(6 + 3 + 3 + 1); // 6 w_in→h, 3 w_h→o, 3 b_h, 1 b_o
        int li = 0;

        // find max |grad| to scale color intensity
        float maxGrad = 1e-6f;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < h; j++)
                maxGrad = Mathf.Max(maxGrad, Mathf.Abs(mlp.Ls[0].dW[i, j]));
        for (int j = 0; j < h; j++)
            maxGrad = Mathf.Max(maxGrad, Mathf.Abs(mlp.Ls[1].dW[j, 0]));
        for (int j = 0; j < h; j++)
            maxGrad = Mathf.Max(maxGrad, Mathf.Abs(mlp.Ls[0].db[j]));
        maxGrad = Mathf.Max(maxGrad, Mathf.Abs(mlp.Ls[1].db[0]));

        // input->hidden weights (2x3)
        if (showWeights)
        {
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < h; j++)
                {
                    var Wval = mlp.Ls[0].W[i, j];
                    var Gval = mlp.Ls[0].dW[i, j];
                    Vector2 mid = (posIn[i] + posH[j]) * 0.5f + PerpOffset(posIn[i], posH[j], 10f);
                    PlaceLabel(labels[li++], mid, $"w{i}{j} {Fmt(Wval)}\n g {Fmt(Gval)}", ColorForGrad(Gval, maxGrad));
                }
        }

        // hidden->output weights (3x1)
        if (showWeights)
        {
            for (int j = 0; j < h; j++)
            {
                var Wval = mlp.Ls[1].W[j, 0];
                var Gval = mlp.Ls[1].dW[j, 0];
                Vector2 mid = (posH[j] + posO) * 0.5f + PerpOffset(posH[j], posO, 10f);
                PlaceLabel(labels[li++], mid, $"v{j} {Fmt(Wval)}\n g {Fmt(Gval)}", ColorForGrad(Gval, maxGrad));
            }
        }

        // hidden biases (3)
        if (showBiases)
        {
            for (int j = 0; j < h; j++)
            {
                var b = mlp.Ls[0].b[j];
                var g = mlp.Ls[0].db[j];
                Vector2 p = posH[j] + new Vector2(28, 0);
                PlaceLabel(labels[li++], p, $"b{j} {Fmt(b)}\n g {Fmt(g)}", ColorForGrad(g, maxGrad));
            }
        }

        // output bias (1)
        if (showBiases)
        {
            var b = mlp.Ls[1].b[0];
            var g = mlp.Ls[1].db[0];
            Vector2 p = posO + new Vector2(30, 0);
            PlaceLabel(labels[li++], p, $"c {Fmt(b)}\n g {Fmt(g)}", ColorForGrad(g, maxGrad));
        }

        tex.Apply(false);
    }

    // ---------- helpers ----------
    Vector2 ToUI(Vector2 pix)
    {
        Vector2 size = rt.rect.size;
        float x = (pix.x / W - 0.5f) * size.x;
        float y = (pix.y / H - 0.5f) * size.y;
        return new Vector2(x, y);
    }

    void EnsureLabelPool(int n)
    {
        while (labels.Count < n)
        {
            var go = new GameObject("ParamLabel", typeof(RectTransform));
            go.transform.SetParent(transform, false);
            var t = go.AddComponent<TextMeshProUGUI>();
            t.raycastTarget = false;
            t.fontSize = fontSize;
            t.alignment = TextAlignmentOptions.Center;
            t.enableWordWrapping = false;
            t.color = new Color(0.9f, 0.9f, 0.9f, 0.95f);
            labels.Add(t);
        }
        for (int i = 0; i < labels.Count; i++)
            labels[i].gameObject.SetActive(i < n);
    }

    void PlaceLabel(TextMeshProUGUI t, Vector2 pixPos, string text, Color c)
    {
        var rtLabel = (RectTransform)t.transform;
        rtLabel.anchoredPosition = ToUI(pixPos);
        t.text = text;
        t.color = c;
    }

    Color ColorForGrad(float g, float maxAbs)
    {
        float a = Mathf.Clamp01(Mathf.Abs(g) / Mathf.Max(1e-6f, maxAbs));
        var baseCol = g >= 0f ? gradPos : gradNeg;
        return Color.Lerp(new Color(0.8f, 0.8f, 0.8f, 0.9f), baseCol, Mathf.Sqrt(a));
    }

    static Vector2 PerpOffset(Vector2 a, Vector2 b, float px)
    {
        Vector2 d = (b - a).normalized;
        Vector2 p = new Vector2(-d.y, d.x); // perpendicular
        return p * px;
    }

    // ----- drawing (pixel) -----
    void Clear()
    {
        var cols = tex.GetPixels32();
        var c = new Color32(18, 18, 18, 255);
        for (int i = 0; i < cols.Length; i++) cols[i] = c;
        tex.SetPixels32(cols);
    }
    void DrawNode(Vector2 c, int rOuter, int rInner, float intensity)
    {
        DrawCircle((int)c.x, (int)c.y, rOuter, ring);
        Color fill = Color.Lerp(new Color(0.2f, 0.2f, 0.2f, 1), fillBase, Mathf.Clamp01(intensity));
        DrawCircleFilled((int)c.x, (int)c.y, rInner, fill);
    }
    void DrawLine(Vector2 a, Vector2 b, Color c) { DrawLine((int)a.x, (int)a.y, (int)b.x, (int)b.y, c); }
    void DrawLine(int x0, int y0, int x1, int y1, Color c)
    {
        int dx = Mathf.Abs(x1 - x0), dy = Mathf.Abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx - dy;
        while (true)
        {
            tex.SetPixel(x0, y0, c);
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
    }
    void DrawCircle(int cx, int cy, int r, Color c)
    {
        int x = r, y = 0, err = 1 - x;
        while (x >= y)
        {
            Set4(cx, cy, x, y, c); Set4(cx, cy, y, x, c);
            y++;
            if (err < 0) err += 2 * y + 1;
            else { x--; err += 2 * (y - x + 1); }
        }
    }
    void DrawCircleFilled(int cx, int cy, int r, Color c)
    {
        for (int y = -r; y <= r; y++)
        {
            int hh = (int)Mathf.Sqrt(r * r - y * y);
            for (int x = -hh; x <= hh; x++) tex.SetPixel(cx + x, cy + y, c);
        }
    }
    void Set4(int cx, int cy, int x, int y, Color c)
    {
        tex.SetPixel(cx + x, cy + y, c);
        tex.SetPixel(cx - x, cy + y, c);
        tex.SetPixel(cx + x, cy - y, c);
        tex.SetPixel(cx - x, cy - y, c);
    }

    // ----- math (mirrors your MLP) -----
    float[] ForwardHidden(MLP mlp, float x0, float x1)
    {
        var L0 = mlp.Ls[0];
        float[] z = new float[L0.b.Length];
        for (int j = 0; j < z.Length; j++)
            z[j] = x0 * L0.W[0, j] + x1 * L0.W[1, j] + L0.b[j];
        var (phi, _) = Activations.Get(mlp.activation);
        for (int j = 0; j < z.Length; j++) z[j] = phi(z[j]);
        return z;
    }
    float ForwardProb(MLP mlp, float[] a0)
    {
        var L1 = mlp.Ls[1];
        float z1 = 0f;
        for (int j = 0; j < a0.Length; j++) z1 += a0[j] * L1.W[j, 0];
        z1 += L1.b[0];
        return 1f / (1f + Mathf.Exp(-z1));
    }
    float NormalizeAct(float a, Act act)
    {
        if (act == Act.Tanh) return 0.5f * (a + 1f);
        if (act == Act.Sigmoid) return a;
        return 1f - Mathf.Exp(-Mathf.Clamp(a, 0f, 10f)); // ReLU
    }

    string Fmt(float v) => v.ToString("+0.000;-0.000;+0.000");
}
