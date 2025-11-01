using UnityEngine;
using UnityEngine.EventSystems;

public class UIButtonSFX : MonoBehaviour, IPointerEnterHandler, IPointerClickHandler
{
    public AudioClip hoverClip;
    public AudioClip clickClip;

    public void OnPointerEnter(PointerEventData eventData)
    {
        if (hoverClip) AudioManager.Instance?.PlaySFX(hoverClip, 0.8f);
    }

    public void OnPointerClick(PointerEventData eventData)
    {
        if (clickClip) AudioManager.Instance?.PlaySFX(clickClip, 1f);
    }

    // If you prefer the Button’s OnClick event in Inspector:
    public void PlayClick()
    {
        if (clickClip) AudioManager.Instance?.PlaySFX(clickClip, 1f);
    }
}
