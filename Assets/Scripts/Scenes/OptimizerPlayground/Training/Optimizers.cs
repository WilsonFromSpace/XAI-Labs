using System;

public interface IOptimizer
{
    string Name { get; }
    void Reset(Layer[] layers);                                 // clear state
    void Apply(Layer[] layers, float lr, int batchSize);        // W -= update(dW), b -= update(db)
}

// ------------------ SGD ------------------
public class OptSGD : IOptimizer
{
    public string Name => "SGD";
    public void Reset(Layer[] layers) { /* no state */ }
    public void Apply(Layer[] layers, float lr, int batchSize)
    {
        float inv = 1f / Math.Max(1, batchSize);
        foreach (var L in layers)
        {
            TinyTensor.AddInPlace(L.W, L.dW, -lr * inv);
            TinyTensor.AddInPlace(L.b, L.db, -lr * inv);
        }
    }
}

// ---------------- Momentum (Polyak) ----------------
public class OptMomentum : IOptimizer
{
    public string Name => "Momentum";
    public float beta = 0.9f;

    float[][,] vW;
    float[][] vb;

    public void Reset(Layer[] layers)
    {
        vW = new float[layers.Length][,];
        vb = new float[layers.Length][];
        for (int k = 0; k < layers.Length; k++)
        {
            vW[k] = new float[layers[k].W.GetLength(0), layers[k].W.GetLength(1)];
            vb[k] = new float[layers[k].b.Length];
        }
    }

    public void Apply(Layer[] layers, float lr, int batchSize)
    {
        if (vW == null || vW.Length != layers.Length) Reset(layers);
        float inv = 1f / Math.Max(1, batchSize);

        for (int k = 0; k < layers.Length; k++)
        {
            var L = layers[k];
            // v = β v + (1-β) g
            for (int i = 0; i < L.W.GetLength(0); i++)
                for (int j = 0; j < L.W.GetLength(1); j++)
                    vW[k][i, j] = beta * vW[k][i, j] + (1f - beta) * (L.dW[i, j] * inv);

            for (int j = 0; j < L.b.Length; j++)
                vb[k][j] = beta * vb[k][j] + (1f - beta) * (L.db[j] * inv);

            // W -= lr * v
            TinyTensor.AddInPlace(L.W, vW[k], -lr);
            TinyTensor.AddInPlace(L.b, vb[k], -lr);
        }
    }
}

// ------------------ Adam ------------------
public class OptAdam : IOptimizer
{
    public string Name => "Adam";
    public float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    int t = 0;

    float[][,] mW, vW;
    float[][] mb, vb;

    public void Reset(Layer[] layers)
    {
        t = 0;
        mW = new float[layers.Length][,];
        vW = new float[layers.Length][,];
        mb = new float[layers.Length][];
        vb = new float[layers.Length][];
        for (int k = 0; k < layers.Length; k++)
        {
            mW[k] = new float[layers[k].W.GetLength(0), layers[k].W.GetLength(1)];
            vW[k] = new float[layers[k].W.GetLength(0), layers[k].W.GetLength(1)];
            mb[k] = new float[layers[k].b.Length];
            vb[k] = new float[layers[k].b.Length];
        }
    }

    public void Apply(Layer[] layers, float lr, int batchSize)
    {
        if (mW == null || mW.Length != layers.Length) Reset(layers);
        t++;
        float inv = 1f / Math.Max(1, batchSize);

        float b1t = (float)Math.Pow(beta1, t);
        float b2t = (float)Math.Pow(beta2, t);
        float corr1 = 1f / (1f - b1t);
        float corr2 = 1f / (1f - b2t);

        for (int k = 0; k < layers.Length; k++)
        {
            var L = layers[k];

            for (int i = 0; i < L.W.GetLength(0); i++)
                for (int j = 0; j < L.W.GetLength(1); j++)
                {
                    float g = L.dW[i, j] * inv;
                    mW[k][i, j] = beta1 * mW[k][i, j] + (1f - beta1) * g;
                    vW[k][i, j] = beta2 * vW[k][i, j] + (1f - beta2) * g * g;

                    float mhat = mW[k][i, j] * corr1;
                    float vhat = vW[k][i, j] * corr2;
                    L.W[i, j] -= lr * mhat / (float)(Math.Sqrt(vhat) + eps);
                }

            for (int j = 0; j < L.b.Length; j++)
            {
                float g = L.db[j] * inv;
                mb[k][j] = beta1 * mb[k][j] + (1f - beta1) * g;
                vb[k][j] = beta2 * vb[k][j] + (1f - beta2) * g * g;

                float mhat = mb[k][j] * corr1;
                float vhat = vb[k][j] * corr2;
                L.b[j] -= lr * mhat / (float)(Math.Sqrt(vhat) + eps);
            }
        }
    }
}
