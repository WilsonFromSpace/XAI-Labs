using UnityEngine;

[CreateAssetMenu(fileName = "Dataset2D_S6", menuName = "XAI/Dataset2D_S6")]
public class Dataset2D_S6 : ScriptableObject
{
    public enum Shape { Blobs, Moons, Rings, Spiral, XOR }

    [Header("Current data")]
    public Vector2[] points;
    public float[] labels;   // 0 or 1
    public int count;

    public void Generate(Shape shape, int n = 600, float noise = 0.1f, float overlap = 0.3f, int seed = 123)
    {
        Random.InitState(seed);
        n = Mathf.Max(50, n);
        points = new Vector2[n];
        labels = new float[n];
        count = n;

        switch (shape)
        {
            case Shape.Blobs: GenBlobs(n, noise, overlap); break;
            case Shape.Moons: GenMoons(n, noise, overlap); break;
            case Shape.Rings: GenRings(n, noise, overlap); break;
            case Shape.Spiral: GenSpiral(n, noise); break;
            case Shape.XOR: GenXOR(n, noise, overlap); break;
        }
    }

    public float[,] XMatrix() { var X = new float[count, 2]; for (int i = 0; i < count; i++) { X[i, 0] = points[i].x; X[i, 1] = points[i].y; } return X; }
    public float[,] YMatrix() { var Y = new float[count, 1]; for (int i = 0; i < count; i++) Y[i, 0] = labels[i]; return Y; }

    // ---------- shapes ----------
    void GenBlobs(int n, float noise, float overlap)
    {
        int n0 = n / 2, n1 = n - n0;
        Vector2 c0 = new Vector2(-0.7f, 0f), c1 = new Vector2(0.7f, 0f);
        float spread = Mathf.Lerp(0.2f, 0.6f, overlap);
        for (int i = 0; i < n0; i++) { points[i] = c0 + spread * RandN2() + noise * RandN2(); labels[i] = 0; }
        for (int i = 0; i < n1; i++) { points[n0 + i] = c1 + spread * RandN2() + noise * RandN2(); labels[n0 + i] = 1; }
        Normalize();
    }
    void GenMoons(int n, float noise, float overlap)
    {
        int n0 = n / 2, n1 = n - n0; float gap = Mathf.Lerp(0.4f, -0.1f, overlap);
        for (int i = 0; i < n0; i++) { float t = Random.value * Mathf.PI; Vector2 p = new Vector2(Mathf.Cos(t), Mathf.Sin(t)); points[i] = p + noise * RandN2(); labels[i] = 0; }
        for (int i = 0; i < n1; i++) { float t = Random.value * Mathf.PI; Vector2 p = new Vector2(1f - Mathf.Cos(t), 1.0f - Mathf.Sin(t) + gap); points[n0 + i] = p + noise * RandN2(); labels[n0 + i] = 1; }
        Normalize();
    }
    void GenRings(int n, float noise, float overlap)
    {
        int n0 = n / 2, n1 = n - n0;
        float r0 = 0.6f, r1 = Mathf.Lerp(1.2f, 0.9f, overlap);
        for (int i = 0; i < n0; i++) { float t = Random.value * 2 * Mathf.PI; points[i] = r0 * new Vector2(Mathf.Cos(t), Mathf.Sin(t)) + noise * RandN2(); labels[i] = 0; }
        for (int i = 0; i < n1; i++) { float t = Random.value * 2 * Mathf.PI; points[n0 + i] = r1 * new Vector2(Mathf.Cos(t), Mathf.Sin(t)) + noise * RandN2(); labels[n0 + i] = 1; }
        Normalize();
    }
    void GenSpiral(int n, float noise)
    {
        int n0 = n / 2, n1 = n - n0; float a = 0.2f, b = 0.9f;
        for (int i = 0; i < n0; i++) { float t = i / (float)n0 * 3.5f * Mathf.PI; float r = a + b * t / (3.5f * Mathf.PI); points[i] = r * new Vector2(Mathf.Cos(t), Mathf.Sin(t)) + noise * RandN2(); labels[i] = 0; }
        for (int i = 0; i < n1; i++) { float t = i / (float)n1 * 3.5f * Mathf.PI + Mathf.PI; float r = a + b * t / (3.5f * Mathf.PI); points[n0 + i] = r * new Vector2(Mathf.Cos(t), Mathf.Sin(t)) + noise * RandN2(); labels[n0 + i] = 1; }
        Normalize();
    }
    void GenXOR(int n, float noise, float overlap)
    {
        float s = Mathf.Lerp(0.25f, 0.6f, overlap);
        for (int i = 0; i < n; i++)
        {
            Vector2 c = new Vector2(Random.value < 0.5f ? -1 : 1, Random.value < 0.5f ? -1 : 1);
            Vector2 p = c + s * RandN2() + noise * RandN2();
            points[i] = p; labels[i] = ((p.x > 0f) ^ (p.y > 0f)) ? 1 : 0;
        }
        Normalize();
    }

    Vector2 RandN2()
    {
        float u = Random.value, v = Random.value;
        float r = Mathf.Sqrt(-2f * Mathf.Log(1 - u)); float th = 2 * Mathf.PI * v;
        return new Vector2(r * Mathf.Cos(th), r * Mathf.Sin(th));
    }

    void Normalize()
    {
        Vector2 min = points[0], max = points[0];
        for (int i = 1; i < points.Length; i++) { min = Vector2.Min(min, points[i]); max = Vector2.Max(max, points[i]); }
        Vector2 c = 0.5f * (min + max);
        Vector2 size = max - min;
        float s = 1.1f / Mathf.Max(size.x, size.y);
        for (int i = 0; i < points.Length; i++) points[i] = (points[i] - c) * s;
    }
}
