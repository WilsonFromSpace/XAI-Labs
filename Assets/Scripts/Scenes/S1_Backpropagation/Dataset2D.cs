using UnityEngine;

[CreateAssetMenu(menuName = "XAI/Dataset2D")]
public class Dataset2D : ScriptableObject
{
    [Header("Counts & Seeds")]
    public int count = 400;
    public int seed = 42;

    [Header("Blob Shape")]
    [Tooltip("Standard deviation of the Gaussian (base). Smaller = tighter blobs.")]
    public float sigma = 0.45f;
    [Tooltip("Multiply Y-std by this to make ellipses. 1 = round.")]
    public float ellipseY = 1f;
    [Tooltip("Clip the Gaussian to avoid far outliers (e.g., 2.5).")]
    public float truncateSigma = 2.5f;

    [Header("Separation & Placement")]
    [Tooltip("Distance between the two blob centers.")]
    public float separation = 3.0f;
    [Tooltip("Allow random rotation of the two centers around the origin.")]
    public bool randomOrientation = true;
    [Tooltip("Allow shifting the mid-point randomly so blobs appear anywhere.")]
    public bool randomTranslation = true;
    [Tooltip("How far from the origin we may shift the mid-point (world units).")]
    public float translationRange = 2.0f;

    [Header("(Optional) World Bounds for safety")]
    public Vector2 worldMin = new Vector2(-5, -5);
    public Vector2 worldMax = new Vector2(5, 5);

    [HideInInspector] public Vector2[] points;
    [HideInInspector] public float[] labels; // 0 or 1

    // --- Public APIs you already call ---
    public void GenerateBlobs() => GenerateBlobsClean(seed);
    public void GenerateBlobs(int? overrideSeed)
    {
        if (overrideSeed.HasValue) seed = overrideSeed.Value;
        GenerateBlobsClean(seed);
    }
    public void ReseedAndGenerate() { seed = Random.Range(1, int.MaxValue); GenerateBlobsClean(seed); }
    // Convenience for the B-scene:
    public void ReseedAndGenerateClean() => ReseedAndGenerate();

    // Matrix helpers
    public float[,] XMatrix()
    {
        var X = new float[count, 2];
        for (int i = 0; i < count; i++) { X[i, 0] = points[i].x; X[i, 1] = points[i].y; }
        return X;
    }
    public float[,] YMatrix()
    {
        var Y = new float[count, 1];
        for (int i = 0; i < count; i++) Y[i, 0] = labels[i];
        return Y;
    }

    // --- New clean generator (truncated Gaussian + random pose) ---
    void GenerateBlobsClean(int s)
    {
        var rnd = new System.Random(s);

        // 1) Choose axis direction (random angle) and mid-point shift (optional)
        float theta = randomOrientation ? (float)(rnd.NextDouble() * Mathf.PI * 2f) : 0f;
        Vector2 dir = new Vector2(Mathf.Cos(theta), Mathf.Sin(theta)); // unit axis
        Vector2 mid = randomTranslation
            ? new Vector2(RandRange(rnd, -translationRange, translationRange),
                          RandRange(rnd, -translationRange, translationRange))
            : Vector2.zero;

        // 2) Two centers along that axis
        Vector2 c0 = mid - dir * (separation * 0.5f);
        Vector2 c1 = mid + dir * (separation * 0.5f);

        // keep inside loose bounds
        c0 = ClampToBounds(c0);
        c1 = ClampToBounds(c1);

        // 3) Sample truncated Gaussian around each center
        points = new Vector2[count];
        labels = new float[count];

        for (int i = 0; i < count; i++)
        {
            bool cls1 = (i % 2 == 0);
            Vector2 c = cls1 ? c1 : c0;

            Vector2 g = TruncatedGaussian2D(rnd, sigma, sigma * ellipseY, truncateSigma);
            // Optional: rotate the local scatter a little around the same theta to align ellipses
            g = Rotate(g, theta * 0.0f); // set to 1.0f if you want ellipses aligned to axis

            points[i] = c + g;
            labels[i] = cls1 ? 1f : 0f;
        }
    }

    // --- Helpers ---
    static float RandRange(System.Random r, float a, float b) => a + (float)r.NextDouble() * (b - a);

    static Vector2 ClampToBounds(Vector2 v)
    {
        return new Vector2(
            Mathf.Clamp(v.x, -4.0f, 4.0f), // a bit inside to avoid touching edges
            Mathf.Clamp(v.y, -4.0f, 4.0f)
        );
    }

    static Vector2 Rotate(Vector2 v, float ang)
    {
        float ca = Mathf.Cos(ang), sa = Mathf.Sin(ang);
        return new Vector2(ca * v.x - sa * v.y, sa * v.x + ca * v.y);
    }

    // Truncated Gaussian via Box–Muller; resample until within k·sigma on each axis
    static Vector2 TruncatedGaussian2D(System.Random rnd, float sx, float sy, float k)
    {
        for (; ; )
        {
            // Box–Muller
            float u1 = 1f - (float)rnd.NextDouble();
            float u2 = 1f - (float)rnd.NextDouble();
            float r = Mathf.Sqrt(-2f * Mathf.Log(u1));
            float z0 = r * Mathf.Cos(2f * Mathf.PI * u2);
            float z1 = r * Mathf.Sin(2f * Mathf.PI * u2);

            float x = z0 * sx;
            float y = z1 * sy;

            if (Mathf.Abs(x) <= k * sx && Mathf.Abs(y) <= k * sy)
                return new Vector2(x, y);
        }
    }
}
