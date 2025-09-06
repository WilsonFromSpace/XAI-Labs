using UnityEngine;
using UnityEngine.UI;

/// Data-only difficulty proxy: average fraction of k nearest neighbors from opposite class.
public class AmbiguityMeterPanel : MonoBehaviour
{
    public RawImage img;
    public Color bg = new(0.08f, 0.08f, 0.1f, 1f);
    public Color lo = new(0.55f, 0.85f, 0.65f, 1f), hi = new(0.95f, 0.55f, 0.45f, 1f);
    public int k = 8;

    Texture2D tex; const int W = 260, H = 36;

    void Awake()
    {
        if (!img) img = GetComponent<RawImage>();
        tex = new Texture2D(W, H, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp }; img.texture = tex;
    }

    public void Redraw(Vector2[] pts, int[] y)
    {
        float amb = ComputeAmbiguity(pts, y, k); // 0..1
        DrawBar(Mathf.Clamp01(amb));
    }

    float ComputeAmbiguity(Vector2[] pts, int[] y, int k)
    {
        if (pts == null || y == null || pts.Length < k + 1) return 0f;
        int n = pts.Length; float sum = 0f;
        for (int i = 0; i < n; i++)
        {
            // naive kNN
            System.Span<float> dist = stackalloc float[128]; // small fast path
            float[] darr = dist.Length >= n ? null : new float[n];
            float[] d = darr ?? new float[n];
            for (int j = 0; j < n; j++) d[j] = (pts[i] - pts[j]).sqrMagnitude + (i == j ? 1e9f : 0f);
            // pick k min
            int opp = 0;
            for (int t = 0; t < k; t++)
            {
                int argmin = 0; float best = 1e9f;
                for (int j = 0; j < n; j++) if (d[j] < best) { best = d[j]; argmin = j; }
                if (y[argmin] != y[i]) opp++;
                d[argmin] = 1e9f;
            }
            sum += opp / (float)k;
        }
        return sum / n;
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
