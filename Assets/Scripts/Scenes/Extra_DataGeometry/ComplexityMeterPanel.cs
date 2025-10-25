using UnityEngine;
using UnityEngine.UI;

/// Model-only boundary complexity: integral of |∇p| over grid (normalized).
public class ComplexityMeterPanel : MonoBehaviour
{
    public RawImage img;
    public Color bg = new(0.08f, 0.08f, 0.1f, 1f);
    public Color lo = new(0.6f, 0.75f, 1f, 1f), hi = new(1f, 0.6f, 0.85f, 1f);
    public Vector2 worldMin = new(-1.2f, -1.2f), worldMax = new(1.2f, 1.2f);
    [Range(48, 256)] public int res = 96;

    Texture2D tex; const int W = 260, H = 36;
    void Awake()
    {
        if (!img) img = GetComponent<RawImage>();
        tex = new Texture2D(W, H, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp }; img.texture = tex;
    }

    public void Configure(Vector2 min, Vector2 max) { worldMin = min; worldMax = max; }

    public void Redraw(MLP_Capacity mlp)
    {
        if (mlp == null) return;
        float v = Mathf.Clamp01(EstimateComplexity(mlp));
        DrawBar(v);
    }

    float EstimateComplexity(MLP_Capacity mlp)
    {
        int R = res;
        float[,] X = new float[R * R, 2];
        int k = 0; for (int y = 0; y < R; y++)
        {
            float wy = Mathf.Lerp(worldMin.y, worldMax.y, y / (R - 1f));
            for (int x = 0; x < R; x++)
            {
                float wx = Mathf.Lerp(worldMin.x, worldMax.x, x / (R - 1f));
                X[k, 0] = wx; X[k, 1] = wy; k++;
            }
        }
        var P = mlp.Forward(X, null, train: false).pred;

        // reshape and finite diff
        float s = 0f; k = 0;
        for (int y = 0; y < R; y++)
        {
            for (int x = 0; x < R; x++, k++)
            {
                float p = P[k, 0];
                float px = (x + 1 < R) ? P[k + 1, 0] - p : 0f;
                float py = (y + 1 < R) ? P[k + R, 0] - p : 0f;
                s += Mathf.Sqrt(px * px + py * py);
            }
        }
        // normalize to ~0..1 across typical settings
        float norm = R * R * 0.02f;
        return Mathf.Clamp01(s / Mathf.Max(1e-6f, norm));
    }

    void DrawBar(float v)
    {
        var px = new Color32[W * H]; var bgc = (Color32)bg; for (int i = 0; i < px.Length; i++) px[i] = bgc;
        int w = Mathf.RoundToInt(v * (W - 2));
        Color c = Color.Lerp(lo, hi, v);
        for (int x = 1; x <= w; x++) for (int y = 1; y < H - 1; y++) px[y * W + x] = (Color32)c;
        // border
        for (int x = 0; x < W; x++) { px[x] = (Color32)Color.white; px[(H - 1) * W + x] = (Color32)new Color(1, 1, 1, 0.25f); }
        for (int y = 0; y < H; y++) { px[y * W] = (Color32)Color.white; px[y * W + (W - 1)] = (Color32)new Color(1, 1, 1, 0.25f); }
        tex.SetPixels32(px); tex.Apply(false);
    }
}
