using UnityEngine;
using UnityEngine.UI;

public class ROCPanel : MonoBehaviour
{
    public RawImage img;
    public Color bg = new Color(0.1f, 0.1f, 0.1f, 1f);
    public Color grid = new Color(0.35f, 0.35f, 0.35f, 0.6f);
    public Color curve = new Color(0.8f, 0.8f, 1f, 1f);
    public Color dot = Color.white;

    Texture2D tex;
    const int W = 220, H = 220;

    void Awake()
    {
        if (!img) img = GetComponent<RawImage>();
        tex = new Texture2D(W, H, TextureFormat.RGBA32, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        img.texture = tex;
    }

    public void Redraw(float[,] P, float[,] Y, float thr)
    {
        // background
        var px = new Color32[W * H];
        var bgc = (Color32)bg;
        for (int i = 0; i < px.Length; i++) px[i] = bgc;
        tex.SetPixels32(px);

        // axes/grid
        DrawLine(0, H - 1, W - 1, H - 1, grid); // x-axis
        DrawLine(0, 0, 0, H - 1, grid);         // y-axis
        for (int i = 1; i < 5; i++)
        {
            int x = Mathf.RoundToInt(i * 0.2f * (W - 1));
            int y = Mathf.RoundToInt(i * 0.2f * (H - 1));
            DrawLine(x, 0, x, H - 1, new Color(grid.r, grid.g, grid.b, 0.35f));
            DrawLine(0, y, W - 1, y, new Color(grid.r, grid.g, grid.b, 0.35f));
        }

        // ROC curve
        Vector2[] roc = ComputeROC(P, Y, 100); // roc[i].x = FPR, roc[i].y = TPR
        Vector2Int? prev = null;
        for (int i = 0; i < roc.Length; i++)
        {
            int x = Mathf.RoundToInt(roc[i].x * (W - 1));
            int y = Mathf.RoundToInt(roc[i].y * (H - 1));
            var p = new Vector2Int(x, y);
            if (prev.HasValue) DrawLine(prev.Value.x, prev.Value.y, p.x, p.y, curve);
            prev = p;
        }

        // operating point for current threshold
        Vector2 pt = PointAtThreshold(P, Y, thr); // (FPR, TPR)
        int xd = Mathf.RoundToInt(pt.x * (W - 1));
        int yd = Mathf.RoundToInt(pt.y * (H - 1));
        DrawDot(xd, yd, 3, dot);

        tex.Apply(false);
    }

    // --- Helpers ---

    Vector2[] ComputeROC(float[,] P, float[,] Y, int steps)
    {
        var r = new Vector2[steps + 1];
        for (int s = 0; s <= steps; s++)
        {
            float t = s / (float)steps;
            r[s] = PointAtThreshold(P, Y, t); // (FPR, TPR)
        }
        return r;
    }

    Vector2 PointAtThreshold(float[,] P, float[,] Y, float thr)
    {
        int N = P.GetLength(0);
        int TP = 0, FP = 0, TN = 0, FN = 0;
        for (int i = 0; i < N; i++)
        {
            int y = Y[i, 0] > 0.5f ? 1 : 0;
            int h = P[i, 0] >= thr ? 1 : 0;
            if (h == 1 && y == 1) TP++;
            else if (h == 1 && y == 0) FP++;
            else if (h == 0 && y == 0) TN++;
            else FN++;
        }
        float tpr = TP + FN == 0 ? 0f : TP / (float)(TP + FN); // recall
        float fpr = FP + TN == 0 ? 0f : FP / (float)(FP + TN);
        return new Vector2(fpr, tpr);
    }

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

    void DrawDot(int x, int y, int r, Color c)
    {
        for (int yy = -r; yy <= r; yy++)
            for (int xx = -r; xx <= r; xx++)
            {
                if (xx * xx + yy * yy <= r * r)
                {
                    int X = x + xx, Yp = y + yy;
                    if (X >= 0 && X < W && Yp >= 0 && Yp < H) tex.SetPixel(X, Yp, c);
                }
            }
    }
}
