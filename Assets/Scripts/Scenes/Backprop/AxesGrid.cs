using UnityEngine;

[ExecuteAlways]
public class AxesGrid : MonoBehaviour
{
    public Color axisColor = new Color(1f, 1f, 1f, 0.5f);
    public Color gridColor = new Color(1f, 1f, 1f, 0.10f);
    public float step = 1f;
    public int extent = 5;

    void OnDrawGizmos()
    {
        Gizmos.color = gridColor;
        for (int i = -extent; i <= extent; i++)
        {
            Gizmos.DrawLine(new Vector3(-extent, i, 0), new Vector3(extent, i, 0));
            Gizmos.DrawLine(new Vector3(i, -extent, 0), new Vector3(i, extent, 0));
        }
        // axes
        Gizmos.color = axisColor;
        Gizmos.DrawLine(new Vector3(-extent, 0, 0), new Vector3(extent, 0, 0));
        Gizmos.DrawLine(new Vector3(0, -extent, 0), new Vector3(0, extent, 0));
    }
}
