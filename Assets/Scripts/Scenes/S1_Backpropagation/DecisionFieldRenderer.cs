using UnityEngine;

[RequireComponent(typeof(SpriteRenderer))]
public class DecisionFieldRenderer : MonoBehaviour
{
    public int texSize = 256;
    public Vector2 worldMin = new Vector2(-5, -5);
    public Vector2 worldMax = new Vector2(5, 5);

    Texture2D tex;
    SpriteRenderer sr;

    void Awake()
    {
        sr = GetComponent<SpriteRenderer>();
        tex = new Texture2D(texSize, texSize, TextureFormat.RGBA32, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        sr.sprite = Sprite.Create(tex, new Rect(0, 0, texSize, texSize), new Vector2(0.5f, 0.5f), texSize);
    }

    public void Redraw(System.Func<Vector2, float> probFunc)
    {
        for (int y = 0; y < texSize; y++)
        {
            float wy = Mathf.Lerp(worldMin.y, worldMax.y, y / (texSize - 1f));
            for (int x = 0; x < texSize; x++)
            {
                float wx = Mathf.Lerp(worldMin.x, worldMax.x, x / (texSize - 1f));
                float p = Mathf.Clamp01(probFunc(new Vector2(wx, wy)));
                // Blue→white→red ramp around 0.5
                Color c = Color.Lerp(Color.blue, Color.red, p);
                // thin contour line near 0.5
                float edge = Mathf.Exp(-Mathf.Pow((p - 0.5f) * 40f, 2f));
                c = Color.Lerp(c, Color.white, edge * 0.4f);
                tex.SetPixel(x, y, c);
            }
        }
        tex.Apply(false);
    }
}
