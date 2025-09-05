using UnityEngine;
using UnityEngine.UI;
public class GapGaugePanel : MonoBehaviour
{
    public RawImage img; public Color bg = new(0.1f, 0.1f, 0.1f, 1f), arc = new(0.6f, 0.85f, 1f, 1f), needle = Color.white;
    Texture2D tex; const int W = 200, H = 120;

    void Awake()
    {
        if (!img) img = GetComponent<RawImage>();
        tex = new Texture2D(W, H, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
        img.texture = tex;
    }

    public void Redraw(float gap)
    {
        var px = new Color32[W * H]; var bgc = (Color32)bg; for (int i = 0; i < px.Length; i++) px[i] = bgc; tex.SetPixels32(px);
        for (int x = 0; x < W; x++)
        {
            float t = x / (W - 1f); float a = Mathf.Lerp(-110f, 110f, t) * Mathf.Deg2Rad;
            int y = H / 2 + Mathf.RoundToInt(Mathf.Sin(a) * (H / 2 - 6)); tex.SetPixel(x, y, arc);
        }
        float g = Mathf.Clamp01(gap / 0.15f); // 0..15% gap
        float ang = Mathf.Lerp(-110f, 110f, g) * Mathf.Deg2Rad;
        int x0 = W / 2, y0 = H - 4, x1 = x0 + Mathf.RoundToInt(Mathf.Sin(ang) * (H - 12)), y1 = y0 - Mathf.RoundToInt(Mathf.Cos(ang) * (H - 12));
        DrawLine(x0, y0, x1, y1, needle);
        tex.Apply(false);
    }

    void DrawLine(int x0, int y0, int x1, int y1, Color c)
    {
        int dx = Mathf.Abs(x1 - x0), dy = Mathf.Abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx - dy;
        while (true)
        {
            tex.SetPixel(x0, y0, c); if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err; if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
    }
}
