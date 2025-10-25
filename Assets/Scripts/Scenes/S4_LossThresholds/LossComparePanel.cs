using UnityEngine;
using UnityEngine.UI;

public class LossComparePanel : MonoBehaviour
{
    public RawImage img;
    public Color bg = new Color(0.1f, 0.1f, 0.1f, 1f);
    public Color barBCE = new Color(1f, 0.85f, 0.6f, 1f);
    public Color barMSE = new Color(0.7f, 0.85f, 1f, 1f);
    public int bins = 20;

    Texture2D tex; const int W = 260, H = 140;

    void Awake() { if (!img) img = GetComponent<RawImage>(); tex = new Texture2D(W, H, TextureFormat.RGBA32, false); tex.wrapMode = TextureWrapMode.Clamp; img.texture = tex; }

    public void Redraw(float[,] P, float[,] Y)
    {
        var px = new Color32[W * H]; var bgc = (Color32)bg; for (int i = 0; i < px.Length; i++) px[i] = bgc; tex.SetPixels32(px);

        int N = P.GetLength(0);
        float[] lBCE = new float[N], lMSE = new float[N];
        for (int i = 0; i < N; i++)
        {
            float p = Mathf.Clamp(P[i, 0], 1e-6f, 1 - 1e-6f);
            float y = Y[i, 0];
            lBCE[i] = -(y * Mathf.Log(p) + (1f - y) * Mathf.Log(1f - p));
            float d = p - y;
            lMSE[i] = 0.5f * d * d;
        }

        float maxB = 0f, maxM = 0f;
        for (int i = 0; i < N; i++) { if (lBCE[i] > maxB) maxB = lBCE[i]; if (lMSE[i] > maxM) maxM = lMSE[i]; }
        float[] hB = Hist(lBCE, bins, 0, Mathf.Max(2.5f, maxB));
        float[] hM = Hist(lMSE, bins, 0, Mathf.Max(0.5f, maxM));

        int half = H / 2;
        DrawBars(hB, 0, half - 2, barBCE);       // BCE top
        DrawBars(hM, half + 2, H - 1, barMSE);     // MSE bottom

        tex.Apply(false);
    }

    float[] Hist(float[] arr, int bins, float min, float max)
    {
        var h = new float[bins];
        float inv = 1f / Mathf.Max(1e-6f, (max - min));
        foreach (var v in arr)
        {
            int b = Mathf.Clamp(Mathf.FloorToInt((v - min) * inv * bins), 0, bins - 1);
            h[b] += 1f;
        }
        float m = 0f; for (int i = 0; i < bins; i++) if (h[i] > m) m = h[i];
        if (m > 0f) for (int i = 0; i < bins; i++) h[i] /= m;
        return h;
    }

    void DrawBars(float[] h, int y0, int y1, Color c)
    {
        int Hseg = y1 - y0 + 1;
        for (int i = 0; i < h.Length; i++)
        {
            int x0 = Mathf.RoundToInt(i * (W - 1) / (float)h.Length);
            int x1 = Mathf.RoundToInt((i + 1) * (W - 1) / (float)h.Length);
            int barH = Mathf.RoundToInt(h[i] * (Hseg - 2));
            for (int x = x0; x <= Mathf.Max(x0, x1 - 1); x++)
                for (int y = y1 - barH; y <= y1; y++)
                    tex.SetPixel(x, y, c);
        }
    }
}
