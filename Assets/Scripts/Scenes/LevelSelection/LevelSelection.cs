using UnityEngine;
using UnityEngine.SceneManagement;

public class LevelSelection: MonoBehaviour
{
    public void LoadLevelS1()
    {
        SceneManager.LoadSceneAsync(2);
    }
    public void LoadLevelS2()
    {
        SceneManager.LoadSceneAsync(3);
    }
    public void LoadLevelS3()
    {
        SceneManager.LoadSceneAsync(4);
    }
    public void LoadLevelS4()
    {
        SceneManager.LoadSceneAsync(5);
    }
    public void LoadLevelS5()
    {
        SceneManager.LoadSceneAsync(6);
    }
    public void LoadLevelS6()
    {
        SceneManager.LoadSceneAsync(7);
    }
    public void LoadLevelExtra1()
    {
        SceneManager.LoadSceneAsync(8);
    }
    public void LoadLevelExtra2()
    {
        SceneManager.LoadSceneAsync(9);
    }

    public void BackToMainMenu()
    {
        SceneManager.LoadSceneAsync(0);
    }
}
