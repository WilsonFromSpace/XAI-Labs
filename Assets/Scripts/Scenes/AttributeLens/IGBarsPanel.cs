using UnityEngine;
using UnityEngine.UI;

public class IGBarsPanel : MonoBehaviour
{
    public RawImage img;
    public Color bg = new Color(0.08f, 0.08f, 0.1f, 1f),
                 cx = new Color(0.9f, 0.7f, 0.4f, 1f),
                 cy = new Color(0.6f, 0.85f, 1f, 1f);
    Texture2D tex; const int W = 180, H = 100;

    void Awake()
    {
        if (!img) img = GetComponent<RawImage>();
        tex = new Texture2D(W, H, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
        img.texture = tex;
    }

    public void Redraw(float igx, float igy)
    {
        var px = new Color32[W * H]; var bgc = (Color32)bg; for (int i = 0; i < px.Length; i++) px[i] = bgc; tex.SetPixels32(px);
        int mid = H / 2; DrawHLine(mid, new Color(0.35f, 0.35f, 0.35f, 0.6f));
        DrawBar(W / 4, igx, cx); DrawBar(3 * W / 4, igy, cy);
        tex.Apply(false);
    }

    public void Clear() { if (tex == null) return; var px = new Color32[W * H]; tex.SetPixels32(px); tex.Apply(false); }

    void DrawBar(int xCenter, float v, Color c)
    {
        int h = Mathf.RoundToInt(Mathf.Clamp(v, -1.2f, 1.2f) * (H * 0.45f));
        int y0 = H / 2, y1 = y0 + h;
        if (y1 < y0) { int t = y0; y0 = y1; y1 = t; }
        for (int x = xCenter - 14; x <= xCenter + 14; x++)
            for (int y = y0; y <= y1; y++)
                if (x >= 0 && x < W && y >= 0 && y < H) tex.SetPixel(x, y, c);
    }

    void DrawHLine(int y, Color c) { for (int x = 0; x < W; x++) tex.SetPixel(x, y, c); }
}
