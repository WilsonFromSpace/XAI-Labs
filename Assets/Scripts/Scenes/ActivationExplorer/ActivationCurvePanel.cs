using UnityEngine;
using UnityEngine.UI;

public class ActivationCurvePanel : MonoBehaviour
{
    public RawImage img;
    public Act act = Act.Tanh;
    Texture2D tex;
    const int W = 300, H = 140;

    void Awake() { if (!img) img = GetComponent<RawImage>(); tex = new Texture2D(W, H, TextureFormat.RGBA32, false); tex.wrapMode = TextureWrapMode.Clamp; img.texture = tex; Redraw(); }

    public void SetAct(Act a) { act = a; Redraw(); }

    void Redraw()
    {
        // bg
        var px = new Color32[W * H];
        for (int i = 0; i < px.Length; i++) px[i] = new Color32(24, 24, 24, 255);
        tex.SetPixels32(px);

        // axes
        DrawLine(0, H / 2, W - 1, H / 2, new Color(0.5f, 0.5f, 0.5f, 0.5f));
        DrawLine(W / 2, 0, W / 2, H - 1, new Color(0.5f, 0.5f, 0.5f, 0.5f));

        var (phi, dphi) = Activations.Get(act);
        // map x∈[-4,4] to pixels
        float Xmin = -4f, Xmax = 4f;
        int ToX(float i) => Mathf.RoundToInt((i - Xmin) / (Xmax - Xmin) * (W - 1));
        int ToY(float y) => Mathf.Clamp(Mathf.RoundToInt((0.5f - y * 0.22f) * (H - 1)), 0, H - 1); // scale to fit

        // φ(x)
        Vector2Int? prev = null;
        for (int i = 0; i < W; i++)
        {
            float x = Mathf.Lerp(Xmin, Xmax, i / (W - 1f));
            float y = phi(x);
            var p = new Vector2Int(i, ToY(y));
            if (prev.HasValue) DrawLine(prev.Value.x, prev.Value.y, p.x, p.y, new Color(1, 0.8f, 0.5f, 1));
            prev = p;
        }
        // φ'(x)
        prev = null;
        for (int i = 0; i < W; i++)
        {
            float x = Mathf.Lerp(Xmin, Xmax, i / (W - 1f));
            float y = dphi(x);
            var p = new Vector2Int(i, ToY(y));
            if (prev.HasValue) DrawLine(prev.Value.x, prev.Value.y, p.x, p.y, new Color(0.7f, 0.85f, 1f, 1));
            prev = p;
        }
        tex.Apply(false);
    }

    void DrawLine(int x0, int y0, int x1, int y1, Color c)
    {
        int dx = Mathf.Abs(x1 - x0), dy = Mathf.Abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx - dy;
        while (true) { tex.SetPixel(x0, y0, c); if (x0 == x1 && y0 == y1) break; int e2 = 2 * err; if (e2 > -dy) { err -= dy; x0 += sx; } if (e2 < dx) { err += dx; y0 += sy; } }
    }
}
