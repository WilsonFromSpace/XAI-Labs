using System;
using UnityEngine;

/// Self-contained MLP for Scene 5 (Capacity Arena).
/// - 1..3 hidden layers (ReLU or Tanh), sigmoid output (binary)
/// - BCE (default) or MSE
/// - Inverted Dropout on hidden layers during training
/// - L1 / L2 regularization applied in StepSGD
///
/// Forward(...) returns (loss, predictions)
public class MLP_Capacity
{
    public enum Act { ReLU, Tanh }
    public enum LossType { BCE, MSE }

    [Serializable]
    public class Layer
    {
        public float[,] W;     // [in, out]
        public float[] b;     // [out]
        // caches
        public float[,] Z;     // [N, out]
        public float[,] A;     // [N, out]
        // grads
        public float[,] dW;    // [in, out]
        public float[] db;    // [out]
        // dropout
        public float[,] dropMask; // [N, out] (null if not used)
    }

    // --- public knobs ---
    public float lr = 0.1f;
    public float l1 = 0f;
    public float l2 = 0f;
    public float dropoutP = 0f;                // 0..0.7 recommended
    public Act activation = Act.ReLU;
    public LossType lossType = LossType.BCE;

    // architecture
    public Layer[] Ls;                          // all layers (last = output)
    System.Random rnd;

    // ctor: input=2, hidden width H, output=1, hidden layers L=1..3
    public MLP_Capacity(int input, int hidden, int output, int layers = 1, int seed = 12345)
    {
        rnd = new System.Random(seed);
        layers = Mathf.Clamp(layers, 1, 3);
        int[] dims = BuildDims(input, hidden, output, layers);

        Ls = new Layer[dims.Length - 1];
        for (int l = 0; l < Ls.Length; l++)
        {
            int fanIn = dims[l];
            int fanOut = dims[l + 1];
            Ls[l] = new Layer
            {
                W = new float[fanIn, fanOut],
                b = new float[fanOut],
                dW = new float[fanIn, fanOut],
                db = new float[fanOut]
            };
            InitWeights(Ls[l].W, l < Ls.Length - 1 ? activation : Act.Tanh); // He/Xavier-ish
        }
    }

    int[] BuildDims(int input, int hidden, int output, int layers)
    {
        // Example for layers=2: [input, H, H, output]
        int[] dims = new int[layers + 2];
        dims[0] = input;
        for (int i = 1; i <= layers; i++) dims[i] = hidden;
        dims[dims.Length - 1] = output;
        return dims;
    }

    void InitWeights(float[,] W, Act actForLayer)
    {
        int fanIn = W.GetLength(0);
        float scale = actForLayer == Act.ReLU
            ? Mathf.Sqrt(2f / Mathf.Max(1, fanIn))
            : Mathf.Sqrt(1f / Mathf.Max(1, fanIn));

        int fin = W.GetLength(0), fout = W.GetLength(1);
        for (int i = 0; i < fin; i++)
            for (int j = 0; j < fout; j++)
                W[i, j] = (float)(NextGaussian() * scale);
    }

    double NextGaussian()
    {
        // Box-Muller
        double u1 = 1.0 - rnd.NextDouble();
        double u2 = 1.0 - rnd.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    // -------------------------- Forward --------------------------
    public (float loss, float[,] pred) Forward(float[,] X, float[,] Y, bool train = false)
    {
        // Hidden layers
        float[,] Aprev = X;
        for (int l = 0; l < Ls.Length - 1; l++)
        {
            var L = Ls[l];
            L.Z = AddBias(MatMul(Aprev, L.W), L.b);
            L.A = Apply(L.Z, activation == Act.ReLU ? (Func<float, float>)ReLU : Tanh);

            // Inverted dropout (train only)
            if (train && dropoutP > 0f)
            {
                int N = L.A.GetLength(0), H = L.A.GetLength(1);
                float keep = 1f - dropoutP;
                if (L.dropMask == null || L.dropMask.GetLength(0) != N || L.dropMask.GetLength(1) != H)
                    L.dropMask = new float[N, H];
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < H; j++)
                    {
                        bool on = UnityEngine.Random.value < keep;
                        L.dropMask[i, j] = on ? (1f / keep) : 0f;
                        L.A[i, j] *= L.dropMask[i, j];
                    }
            }
            else
            {
                L.dropMask = null;
            }

            Aprev = L.A;
        }

        // Output layer (sigmoid)
        var O = Ls[Ls.Length - 1];
        O.Z = AddBias(MatMul(Aprev, O.W), O.b);
        float[,] P = Apply(O.Z, Sigmoid); // N x 1

        // Loss
        float loss = 0f;
        if (Y != null)
        {
            int N = P.GetLength(0);
            if (lossType == LossType.BCE)
            {
                for (int i = 0; i < N; i++)
                {
                    float p = Mathf.Clamp(P[i, 0], 1e-6f, 1f - 1e-6f);
                    float y = Y[i, 0];
                    loss += -(y * Mathf.Log(p) + (1f - y) * Mathf.Log(1f - p));
                }
                loss /= Mathf.Max(1, N);
            }
            else // MSE
            {
                for (int i = 0; i < N; i++)
                {
                    float d = P[i, 0] - Y[i, 0];
                    loss += 0.5f * d * d;
                }
                loss /= Mathf.Max(1, N);
            }
        }

        // Backprop for this forward pass (compute grads)
        if (Y != null)
            Backward(X, Y, P);

        return (loss, P);
    }

    // -------------------------- Backward --------------------------
    void Backward(float[,] X, float[,] Y, float[,] P)
    {
        int N = P.GetLength(0);

        // dZ_out
        float[,] dZ = new float[N, 1];
        if (lossType == LossType.BCE)
        {
            for (int i = 0; i < N; i++) dZ[i, 0] = P[i, 0] - Y[i, 0]; // sigmoid+CE simplification
        }
        else // MSE
        {
            for (int i = 0; i < N; i++)
            {
                float p = P[i, 0], y = Y[i, 0];
                dZ[i, 0] = (p - y) * p * (1f - p); // dL/dp * dp/dz
            }
        }

        // Output layer grads
        int Llast = Ls.Length - 1;
        var Lout = Ls[Llast];
        float[,] Aprev = (Llast == 0) ? X : Ls[Llast - 1].A;

        Lout.dW = Scale(MatMul(Transpose(Aprev), dZ), 1f / Mathf.Max(1, N));
        Lout.db = MeanCols(dZ);

        // Propagate back
        float[,] dAprev = MatMul(dZ, Transpose(Lout.W));

        // Hidden layers (reverse)
        for (int l = Llast - 1; l >= 0; l--)
        {
            var L = Ls[l];

            float[,] dZl = (activation == Act.ReLU)
                ? Hadamard(dAprev, Apply(L.Z, ReLUprime))
                : Hadamard(dAprev, Apply(L.Z, TanhPrime));

            // undo dropout scaling
            if (L.dropMask != null)
                dZl = Hadamard(dZl, L.dropMask);

            float[,] Aprev_l = (l == 0) ? X : Ls[l - 1].A;

            L.dW = Scale(MatMul(Transpose(Aprev_l), dZl), 1f / Mathf.Max(1, N));
            L.db = MeanCols(dZl);

            if (l > 0)
                dAprev = MatMul(dZl, Transpose(L.W));
        }
    }

    // -------------------------- SGD step --------------------------
    public void StepSGD(int batchSize)
    {
        foreach (var L in Ls)
        {
            int fin = L.W.GetLength(0), fout = L.W.GetLength(1);

            for (int i = 0; i < fin; i++)
                for (int j = 0; j < fout; j++)
                {
                    float g = L.dW[i, j];
                    if (l2 > 0f) g += l2 * L.W[i, j];              // L2 decay
                    if (l1 > 0f) g += l1 * Mathf.Sign(L.W[i, j]);  // L1 subgradient
                    L.W[i, j] -= lr * g;
                }

            for (int j = 0; j < fout; j++)
                L.b[j] -= lr * L.db[j];
        }
    }

    // -------------------------- Tiny tensor helpers --------------------------
    static float[,] MatMul(float[,] A, float[,] B)
    {
        int n = A.GetLength(0), d = A.GetLength(1), m = B.GetLength(1);
        float[,] C = new float[n, m];
        for (int i = 0; i < n; i++)
            for (int k = 0; k < d; k++)
            {
                float aik = A[i, k];
                for (int j = 0; j < m; j++) C[i, j] += aik * B[k, j];
            }
        return C;
    }

    static float[,] Transpose(float[,] A)
    {
        int n = A.GetLength(0), m = A.GetLength(1);
        float[,] T = new float[m, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) T[j, i] = A[i, j];
        return T;
    }

    static float[,] AddBias(float[,] X, float[] b)
    {
        int n = X.GetLength(0), m = X.GetLength(1);
        float[,] Y = new float[n, m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) Y[i, j] = X[i, j] + b[j];
        return Y;
    }

    static float[,] Apply(float[,] A, Func<float, float> f)
    {
        int n = A.GetLength(0), m = A.GetLength(1);
        float[,] Y = new float[n, m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) Y[i, j] = f(A[i, j]);
        return Y;
    }

    static float[] MeanCols(float[,] A)
    {
        int n = A.GetLength(0), m = A.GetLength(1);
        float[] v = new float[m];
        if (n == 0) return v;
        float inv = 1f / n;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) v[j] += A[i, j] * inv;
        return v;
    }

    static float[,] Scale(float[,] A, float s)
    {
        int n = A.GetLength(0), m = A.GetLength(1);
        float[,] Y = new float[n, m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) Y[i, j] = A[i, j] * s;
        return Y;
    }

    static float[,] Hadamard(float[,] A, float[,] B)
    {
        int n = A.GetLength(0), m = A.GetLength(1);
        float[,] Y = new float[n, m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) Y[i, j] = A[i, j] * B[i, j];
        return Y;
    }

    // activations
    static float ReLU(float x) => x > 0f ? x : 0f;
    static float ReLUprime(float x) => x > 0f ? 1f : 0f;
    static float Tanh(float x) => (float)Math.Tanh(x);
    static float TanhPrime(float x) { float t = (float)Math.Tanh(x); return 1f - t * t; }
    static float Sigmoid(float x) => 1f / (1f + Mathf.Exp(-x));
}
