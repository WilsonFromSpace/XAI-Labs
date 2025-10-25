using UnityEngine;
using System.Collections.Generic;

[ExecuteAlways]
public class GradientEdgeOverlayB : MonoBehaviour
{
    [Header("Visual")]
    public Sprite nodeSprite;
    public Material lineMaterial;
    public float baseLineWidth = 0.025f;
    public float maxBoost = 6f;
    public Color posColor = new Color(0.95f, 0.35f, 0.35f);
    public Color negColor = new Color(0.35f, 0.55f, 0.95f);
    public Color nodeColor = new Color(1f, 1f, 1f, 0.85f);

    [Header("Layout (world positions)")]
    public Vector2 inA = new Vector2(-4f, +1f);
    public Vector2 inB = new Vector2(-4f, -1f);
    public Vector2 hid0 = new Vector2(0f, +2f);
    public Vector2 hid1 = new Vector2(0f, 0f);
    public Vector2 hid2 = new Vector2(0f, -2f);
    public Vector2 outN = new Vector2(+4f, 0f);

    MLP mlpRef;

    class Edge { public LineRenderer lr; public int i, j; public int layerIdx; }
    class Node { public Transform tf; public int layerIdx; public int j; }
    readonly List<Edge> edges = new List<Edge>();
    readonly List<Node> nodes = new List<Node>();

    void ClearAll()
    {
        foreach (var e in edges) if (e.lr) DestroyImmediate(e.lr.gameObject);
        foreach (var n in nodes) if (n.tf) DestroyImmediate(n.tf.gameObject);
        edges.Clear(); nodes.Clear();
    }

    public void SyncFromMLP(MLP mlp)
    {
        if (mlp == null) return;
        if (mlpRef != mlp) { mlpRef = mlp; BuildOnce(); }
        UpdateVisuals();
    }

    void BuildOnce()
    {
        ClearAll();
        // nodes
        nodes.Add(SpawnNode(inA, -1, 0));
        nodes.Add(SpawnNode(inB, -1, 1));
        nodes.Add(SpawnNode(hid0, 0, 0));
        nodes.Add(SpawnNode(hid1, 0, 1));
        nodes.Add(SpawnNode(hid2, 0, 2));
        nodes.Add(SpawnNode(outN, 1, 0));

        // edges input->hidden (2x3)
        var IN = new Vector2[] { inA, inB };
        var H = new Vector2[] { hid0, hid1, hid2 };
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                edges.Add(SpawnEdge(IN[i], H[j], 0, i, j));

        // edges hidden->output (3x1)
        for (int j = 0; j < 3; j++)
            edges.Add(SpawnEdge(H[j], outN, 1, j, 0));
    }

    Node SpawnNode(Vector2 pos, int layerIdx, int j)
    {
        var go = new GameObject($"Node_{layerIdx}_{j}");
        go.transform.SetParent(transform, false);
        go.transform.position = pos;
        var sr = go.AddComponent<SpriteRenderer>();
        sr.sprite = nodeSprite;
        sr.color = nodeColor;
        sr.sortingOrder = 20;
        go.transform.localScale = Vector3.one * 0.35f;
        return new Node { tf = go.transform, layerIdx = layerIdx, j = j };
    }

    Edge SpawnEdge(Vector2 a, Vector2 b, int layerIdx, int i, int j)
    {
        var go = new GameObject($"Edge_{layerIdx}_{i}_{j}");
        go.transform.SetParent(transform, false);
        var lr = go.AddComponent<LineRenderer>();
        lr.material = lineMaterial;
        lr.positionCount = 2;
        lr.useWorldSpace = true;
        lr.numCapVertices = 8;
        lr.sortingOrder = 15;
        lr.SetPosition(0, a);
        lr.SetPosition(1, b);
        lr.startWidth = lr.endWidth = baseLineWidth;
        return new Edge { lr = lr, i = i, j = j, layerIdx = layerIdx };
    }

    void UpdateVisuals()
    {
        if (mlpRef == null) return;

        float maxAbs = 1e-6f;
        foreach (var e in edges)
        {
            var L = mlpRef.Ls[e.layerIdx];
            float g = Mathf.Abs(L.dW[e.i, e.j]);
            if (g > maxAbs) maxAbs = g;
        }
        foreach (var e in edges)
        {
            var L = mlpRef.Ls[e.layerIdx];
            float g = L.dW[e.i, e.j];
            float a = Mathf.Clamp01(Mathf.Abs(g) / maxAbs);
            float w = baseLineWidth * Mathf.Lerp(1f, maxBoost, a);
            e.lr.startWidth = e.lr.endWidth = w;
            e.lr.startColor = e.lr.endColor = (g >= 0f ? posColor : negColor);
        }

        float maxB = 1e-6f;
        foreach (var n in nodes)
        {
            if (n.layerIdx < 0) continue;
            var L = mlpRef.Ls[n.layerIdx];
            float b = Mathf.Abs(L.db[n.j]);
            if (b > maxB) maxB = b;
        }
        foreach (var n in nodes)
        {
            var sr = n.tf.GetComponent<SpriteRenderer>();
            if (n.layerIdx < 0) { n.tf.localScale = Vector3.one * 0.35f; continue; }
            var L = mlpRef.Ls[n.layerIdx];
            float a = Mathf.Clamp01(Mathf.Abs(L.db[n.j]) / maxB);
            float s = Mathf.Lerp(0.35f, 0.6f, a);
            n.tf.localScale = Vector3.one * s;
        }
    }
}
