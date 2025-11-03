using UnityEngine;

public class AudioManager : MonoBehaviour
{
    public static AudioManager Instance { get; private set; }

    [Header("Assign in Inspector")]
    public AudioSource musicSource;   // looped background music
    public AudioSource sfxSource;     // UI one-shots

    void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject);

        if (musicSource != null)
        {
            musicSource.loop = true;
            // Do not auto-play here; let Scene set the clip (see Step 3).
        }
        if (sfxSource != null)
        {
            sfxSource.loop = false;
            sfxSource.playOnAwake = false;
        }
    }

    // --- MUSIC ---
    public void PlayMusic(AudioClip clip, float volume = 1f)
    {
        if (clip == null || musicSource == null) return;
        if (musicSource.clip == clip && musicSource.isPlaying) return; // already playing
        musicSource.clip = clip;
        musicSource.volume = Mathf.Clamp01(volume);
        musicSource.Play();
    }

    public void StopMusic()
    {
        if (musicSource == null) return;
        musicSource.Stop();
        musicSource.clip = null;
    }

    public void SetMusicVolume(float v01)
    {
        if (musicSource == null) return;
        musicSource.volume = Mathf.Clamp01(v01);
    }

    // --- SFX ---
    public void PlaySFX(AudioClip clip, float volume = 1f, float pitch = 1f)
    {
        if (clip == null || sfxSource == null) return;
        sfxSource.pitch = pitch;
        sfxSource.PlayOneShot(clip, Mathf.Clamp01(volume));
    }

    public void SetSfxVolume(float v01)
    {
        if (sfxSource == null) return;
        sfxSource.volume = Mathf.Clamp01(v01);
    }

    private Coroutine _fadeCo;

    public void PlayMusicFade(AudioClip clip, float fadeDuration = 0.6f, float targetVol = 1f)
    {
        if (clip == null || musicSource == null) return;
        if (_fadeCo != null) StopCoroutine(_fadeCo);
        _fadeCo = StartCoroutine(FadeSwap(clip, fadeDuration, Mathf.Clamp01(targetVol)));
    }

    private System.Collections.IEnumerator FadeSwap(AudioClip newClip, float t, float targetVol)
    {
        float startVol = musicSource.volume;
        // fade out
        for (float x = 0; x < t; x += Time.unscaledDeltaTime)
        {
            musicSource.volume = Mathf.Lerp(startVol, 0f, x / t);
            yield return null;
        }
        musicSource.volume = 0f;
        musicSource.clip = newClip;
        musicSource.loop = true;
        musicSource.Play();
        // fade in
        for (float x = 0; x < t; x += Time.unscaledDeltaTime)
        {
            musicSource.volume = Mathf.Lerp(0f, targetVol, x / t);
            yield return null;
        }
        musicSource.volume = targetVol;
        _fadeCo = null;
    }
}
