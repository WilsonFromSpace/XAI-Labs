using UnityEngine;
using UnityEngine.EventSystems;

public class UIButtonSFX : MonoBehaviour, IPointerEnterHandler, IPointerClickHandler
{
    public AudioClip hoverClip;
    public AudioClip clickClip;
    [Range(0f, 1f)] public float hoverVol = 0.7f;
    [Range(0f, 1f)] public float clickVol = 1f;

    public void OnPointerEnter(PointerEventData e)
    {
        if (hoverClip) AudioManager.Instance?.PlaySFX(hoverClip, hoverVol);
    }
    public void OnPointerClick(PointerEventData e)
    {
        if (clickClip) AudioManager.Instance?.PlaySFX(clickClip, clickVol);
    }
}
