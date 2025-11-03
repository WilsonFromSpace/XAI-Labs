using UnityEngine;

public class SceneMusic : MonoBehaviour
{
    public AudioClip track;                 // set per scene
    [Range(0f, 1f)] public float volume = 0.8f;
    public bool fade = true;
    [Range(0f, 2f)] public float fadeTime = 0.6f;

    void Start()
    {
        if (!track) return;
        if (fade) AudioManager.Instance?.PlayMusicFade(track, fadeTime, volume);
        else AudioManager.Instance?.PlayMusic(track, volume);
    }
}
