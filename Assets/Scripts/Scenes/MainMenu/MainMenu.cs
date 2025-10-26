using UnityEngine;
using UnityEngine.SceneManagement;

public class MainMenu : MonoBehaviour
{
    public void PlayGame()
    {
        SceneManager.LoadSceneAsync(2);
    }
    public void SelectLevel()
    {
        SceneManager.LoadSceneAsync(1);
    }
    
    public void ExitGame()
    {
        Application.Quit();
        Debug.Log("Quit!");
    }
}
