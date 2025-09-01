using System;

[Serializable]
public class Layer
{
    public float[,] W;
    public float[] b;

    // caches
    public float[,] X, Z, A;
    public float[,] dW;
    public float[] db;
}

public class MLP
{
    public Layer[] Ls;
    public Act activation = Act.Tanh;      // used for HIDDEN layers
    public LossType lossType = LossType.BCE;
    public float lr = 0.05f;

    readonly Random rnd;

    public MLP(int input, int hidden, int output, int seed = 123)
    {
        rnd = new Random(seed);
        Ls = new Layer[2];
        Ls[0] = new Layer { W = RandInit(input, hidden, hidden, hiddenAct: true), b = new float[hidden] };
        Ls[1] = new Layer { W = RandInit(hidden, output, output, hiddenAct: false), b = new float[output] };
    }

    float[,] RandInit(int fanIn, int fanOut, int dummy, bool hiddenAct)
    {
        // He for ReLU; Xavier/Glorot for Tanh/Sigmoid
        double limit;
        if (hiddenAct && activation == Act.ReLU)
            limit = Math.Sqrt(6.0 / fanIn);                 // He-uniform
        else
            limit = Math.Sqrt(6.0 / (fanIn + fanOut));      // Glorot-uniform

        var M = new float[fanIn, fanOut];
        for (int i = 0; i < fanIn; i++)
            for (int j = 0; j < fanOut; j++)
                M[i, j] = (float)((rnd.NextDouble() * 2 - 1) * limit);
        return M;
    }

    public (float loss, float[,] prob) Forward(float[,] X, float[,] Y)
    {
        var (phi, _) = Activations.Get(activation);

        // Hidden layer: affine + activation
        var L0 = Ls[0];
        L0.X = X;
        L0.Z = TinyTensor.AddBiasRow(TinyTensor.MatMul(X, L0.W), L0.b);
        L0.A = TinyTensor.Apply(L0.Z, phi);

        // Output layer: affine ONLY (linear logits)
        var L1 = Ls[1];
        L1.X = L0.A;
        L1.Z = TinyTensor.AddBiasRow(TinyTensor.MatMul(L1.X, L1.W), L1.b);
        L1.A = L1.Z; // identity

        // For BCE we pass probabilities to the loss but return d(logits) = p - y
        float[,] P = L1.A;
        if (lossType == LossType.BCE && P.GetLength(1) == 1)
            P = TinyTensor.Apply(P, z => 1f / (1f + (float)Math.Exp(-z)));

        var (loss, dOut) = (lossType == LossType.MSE)
            ? Losses.MSE(P, Y)     // dOut = dL/dP
            : Losses.BCE(P, Y);    // dOut = (p - y), i.e., dL/d(logit)

        Backward(dOut);
        return (loss, P);
    }

    void Backward(float[,] dOut)
    {
        // Output layer (identity): dZ1 = dOut
        var L1 = Ls[1];
        var dZ1 = dOut;
        L1.dW = TinyTensor.MatMul(TinyTensor.Transpose(L1.X), dZ1);
        L1.db = TinyTensor.ColSum(dZ1);
        var dA0 = TinyTensor.MatMul(dZ1, TinyTensor.Transpose(L1.W));

        // Hidden layer: multiply by activation derivative
        var L0 = Ls[0];
        var (_, dphi) = Activations.Get(activation);
        var dZ0 = TinyTensor.Hadamard(TinyTensor.Apply(L0.Z, dphi), dA0);
        L0.dW = TinyTensor.MatMul(TinyTensor.Transpose(L0.X), dZ0);
        L0.db = TinyTensor.ColSum(dZ0);
    }

    public void StepSGD(int batchSize)
    {
        foreach (var L in Ls)
        {
            var dWn = Scale(L.dW, 1f / Math.Max(1, batchSize));
            var dbn = Scale(L.db, 1f / Math.Max(1, batchSize));
            TinyTensor.AddInPlace(L.W, dWn, -lr);
            TinyTensor.AddInPlace(L.b, dbn, -lr);
        }
    }

    public void ResetWeights(int seed = -1)
    {
        var r = (seed < 0) ? new Random(Guid.NewGuid().GetHashCode()) : new Random(seed);
        // re-init with same criteria
        Ls[0].W = RandInit(Ls[0].W.GetLength(0), Ls[0].W.GetLength(1), 0, hiddenAct: true);
        Ls[0].b = new float[Ls[0].b.Length];
        Ls[1].W = RandInit(Ls[1].W.GetLength(0), Ls[1].W.GetLength(1), 0, hiddenAct: false);
        Ls[1].b = new float[Ls[1].b.Length];
    }

    float[,] Scale(float[,] A, float s)
    {
        int n = A.GetLength(0), m = A.GetLength(1);
        var C = new float[n, m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                C[i, j] = A[i, j] * s;
        return C;
    }
    float[] Scale(float[] a, float s)
    {
        var c = new float[a.Length];
        for (int i = 0; i < a.Length; i++) c[i] = a[i] * s;
        return c;
    }
}
