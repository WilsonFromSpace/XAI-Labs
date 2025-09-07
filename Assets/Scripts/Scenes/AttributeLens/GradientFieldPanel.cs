using UnityEngine;
using UnityEngine.UI;

/// Quiver-like gradient field (sparse ∇p arrows).
public class GradientFieldPanel : MonoBehaviour
{
    public RawImage img;
    public Vector2 worldMin, worldMax;
    public Color bg = new(0, 0, 0, 0), arrow = new(1f, 1f, 1f, 0.35f);
    Texture2D tex; const int W = 320, H = 320;

    void Awake()
    {
        if (!img) img = GetComponent<RawImage>();
        tex = new Texture2D(W, H, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp }; img.texture = tex; Clear();
    }

    public void Configure(Vector2 min, Vector2 max) { worldMin = min; worldMax = max; }

    public void Redraw(MLP_Capacity mlp, int grid = 16)
    {
        Clear();
        float dx = (worldMax.x - worldMin.x) / (grid - 1f), dy = (worldMax.y - worldMin.y) / (grid - 1f);
        for (int gy = 0; gy < grid; gy++)
            for (int gx = 0; gx < grid; gx++)
            {
                float wx = worldMin.x + gx * dx;
                float wy = worldMin.y + gy * dy;
                Vector2 g = Grad(mlp, new Vector2(wx, wy));
                float L = g.magnitude; if (L < 1e-6f) continue;
                Vector2 d = g / L * 6f;                // arrow length in pixels
                int x = Mathf.RoundToInt((wx - worldMin.x) / (worldMax.x - worldMin.x) * (W - 1));
                int y = Mathf.RoundToInt((wy - worldMin.y) / (worldMax.y - worldMin.y) * (H - 1));
                DrawLine(x, y, x + Mathf.RoundToInt(d.x), y + Mathf.RoundToInt(d.y), arrow);
            }
        tex.Apply(false);
    }

    public void Clear() { var px = new Color32[W * H]; tex.SetPixels32(px); tex.Apply(false); }

    Vector2 Grad(MLP_Capacity mlp, Vector2 x)
    {
        var Xin = new float[1, 2]; Xin[0, 0] = x.x; Xin[0, 1] = x.y;
        var (loss, P) = mlp.Forward(Xin, null, false);
        float p = Mathf.Clamp01(P[0, 0]);
        float dpdz = p * (1f - p);
        int Llast = mlp.Ls.Length - 1; var Lout = mlp.Ls[Llast];
        int H = Lout.W.GetLength(0);
        float[] gA = new float[H]; for (int j = 0; j < H; j++) gA[j] = dpdz * Lout.W[j, 0];

        for (int l = Llast - 1; l >= 0; l--)
        {
            var L = mlp.Ls[l]; int outD = L.W.GetLength(1), inD = L.W.GetLength(0);
            float[] actp = new float[outD];
            for (int j = 0; j < outD; j++) { float z = L.Z[0, j]; actp[j] = (mlp.activation == MLP_Capacity.Act.ReLU) ? (z > 0f ? 1f : 0f) : (1f - L.A[0, j] * L.A[0, j]); }
            float[] gZ = new float[outD]; for (int j = 0; j < outD; j++) gZ[j] = gA[j] * actp[j];
            float[] gPrev = new float[inD];
            for (int i = 0; i < inD; i++) for (int j = 0; j < outD; j++) gPrev[i] += gZ[j] * L.W[i, j];
            gA = gPrev; if (l == 0) return new Vector2(gA[0], gA[1]);
        }
        return Vector2.zero;
    }

    void DrawLine(int x0, int y0, int x1, int y1, Color c)
    {
        int dx = Mathf.Abs(x1 - x0), dy = Mathf.Abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx - dy;
        while (true)
        {
            if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) tex.SetPixel(x0, y0, c);
            if (x0 == x1 && y0 == y1) break; int e2 = 2 * err; if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
    }
}
