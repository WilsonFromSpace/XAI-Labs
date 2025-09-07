using UnityEngine;

/// Draws a thin p≈0.5 decision contour behind the points using marching squares.
/// Attach to a world GameObject with a SpriteRenderer (e.g., "DecisionContour").
/// Call Configure(...) once (bounds), then Redraw(mlp) whenever the model updates.
public class DecisionContourPanel : MonoBehaviour
{
    [Header("World bounds (in world units)")]
    public Vector2 worldMin = new Vector2(-1f, -1f);
    public Vector2 worldMax = new Vector2(1f, 1f);

    [Header("Rendering")]
    [Range(48, 256)] public int resolution = 128;                 // grid samples per axis
    [Range(1, 3)] public int lineThickness = 2;
    public Color mainLine = Color.white;                           // p=0.5
    public Color auxLine = new Color(1f, 1f, 1f, 0.25f);             // p=0.25 / 0.75 (optional)

    SpriteRenderer sr;
    Texture2D tex;
    float pxPerUnit = 100f;

    void Awake()
    {
        sr = GetComponent<SpriteRenderer>();
        if (!sr) sr = gameObject.AddComponent<SpriteRenderer>();
        sr.sortingOrder = 1; // dots can use 10 so this stays behind
        EnsureTexture();
    }

    void EnsureTexture()
    {
        int w = resolution, h = resolution;
        if (tex == null || tex.width != w || tex.height != h)
        {
            tex = new Texture2D(w, h, TextureFormat.RGBA32, false);
            tex.wrapMode = TextureWrapMode.Clamp;
            var sp = Sprite.Create(tex, new Rect(0, 0, w, h), new Vector2(0.5f, 0.5f), pxPerUnit);
            sr.sprite = sp;
        }
        float widthU = Mathf.Max(1e-6f, worldMax.x - worldMin.x);
        pxPerUnit = tex.width / widthU;
        var s = Sprite.Create(tex, new Rect(0, 0, tex.width, tex.height), new Vector2(0.5f, 0.5f), pxPerUnit);
        sr.sprite = s;

        Vector2 c = 0.5f * (worldMin + worldMax);
        transform.position = new Vector3(c.x, c.y, 0f);
        transform.localScale = Vector3.one; // PPU handles size
    }

    public void Configure(Vector2 min, Vector2 max)
    {
        worldMin = min; worldMax = max;
        EnsureTexture();
        ClearTex();
        tex.Apply(false);
    }

    public void Redraw(MLP_Capacity mlp)
    {
        if (mlp == null || tex == null) return;   // explicit null check (no !mlp)
        ClearTex();

        int W = tex.width, H = tex.height;
        float[,] X = new float[W * H, 2];
        int k = 0;
        for (int y = 0; y < H; y++)
        {
            float wy = Mathf.Lerp(worldMin.y, worldMax.y, y / (H - 1f));
            for (int x = 0; x < W; x++)
            {
                float wx = Mathf.Lerp(worldMin.x, worldMax.x, x / (W - 1f));
                X[k, 0] = wx; X[k, 1] = wy; k++;
            }
        }
        var preds = mlp.Forward(X, null, train: false).pred;

        float[,] F = new float[W, H];
        k = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++, k++)
                F[x, y] = preds[k, 0];

        DrawIso(F, 0.50f, mainLine, lineThickness);
        DrawIso(F, 0.25f, auxLine, 1);
        DrawIso(F, 0.75f, auxLine, 1);

        tex.Apply(false);
    }

    void DrawIso(float[,] F, float level, Color c, int thick)
    {
        int W = F.GetLength(0), H = F.GetLength(1);
        for (int y = 0; y < H - 1; y++)
        {
            for (int x = 0; x < W - 1; x++)
            {
                float f00 = F[x, y], f10 = F[x + 1, y];
                float f01 = F[x, y + 1], f11 = F[x + 1, y + 1];

                int idx = 0;
                if (f00 > level) idx |= 1;
                if (f10 > level) idx |= 2;
                if (f11 > level) idx |= 4;
                if (f01 > level) idx |= 8;
                if (idx == 0 || idx == 15) continue;

                Vector2 a, b;

                Vector2 eL = new Vector2(x, y + T(level, f00, f01));
                Vector2 eR = new Vector2(x + 1, y + T(level, f10, f11));
                Vector2 eB = new Vector2(x + T(level, f00, f10), y);
                Vector2 eT = new Vector2(x + T(level, f01, f11), y + 1);

                switch (idx)
                {
                    case 1: case 14: a = eB; b = eL; break;
                    case 2: case 13: a = eR; b = eB; break;
                    case 3: case 12: a = eR; b = eL; break;
                    case 4: case 11: a = eT; b = eR; break;
                    case 5: a = eB; b = eT; DrawLine(eL, eR, c, thick); break;
                    case 6: case 9: a = eT; b = eB; break;
                    case 7: case 8: a = eL; b = eT; break;
                    case 10: a = eL; b = eR; break;
                    default: a = eL; b = eR; break;
                }
                DrawLine(a, b, c, thick);
            }
        }
    }

    float T(float level, float a, float b)
    {
        float denom = (b - a);
        if (Mathf.Abs(denom) < 1e-6f) return 0.5f;
        return Mathf.Clamp01((level - a) / denom);
    }

    void DrawLine(Vector2 a, Vector2 b, Color col, int thick)
    {
        int x0 = Mathf.RoundToInt(a.x), y0 = Mathf.RoundToInt(a.y);
        int x1 = Mathf.RoundToInt(b.x), y1 = Mathf.RoundToInt(b.y);
        int dx = Mathf.Abs(x1 - x0), dy = Mathf.Abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx - dy;

        while (true)
        {
            Plot(x0, y0, col, thick);
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
    }

    void Plot(int x, int y, Color col, int r)
    {
        int W = tex.width, H = tex.height;
        for (int yy = -r; yy <= r; yy++)
            for (int xx = -r; xx <= r; xx++)
            {
                if (xx * xx + yy * yy > r * r) continue;
                int X = x + xx, Y = y + yy;
                if (X >= 0 && X < W && Y >= 0 && Y < H)
                    tex.SetPixel(X, Y, col);
            }
    }

    void ClearTex()
    {
        int W = tex.width, H = tex.height;
        var bg = new Color32(0, 0, 0, 0);
        var arr = new Color32[W * H];
        for (int i = 0; i < arr.Length; i++) arr[i] = bg;
        tex.SetPixels32(arr);
    }
}
