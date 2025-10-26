using UnityEngine;
using System.Collections.Generic;

public class NetworkLines : MonoBehaviour
{
    public ParticleSystem particles;
    public Material lineMaterial;
    public float connectDistance = 1.5f;
    public int maxConnections = 3;
    private ParticleSystem.Particle[] particleArray;
    private List<LineRenderer> lines = new();

    void LateUpdate()
    {
        int count = particles.GetParticles(particleArray = new ParticleSystem.Particle[particles.particleCount]);
        foreach (var line in lines) Destroy(line.gameObject);
        lines.Clear();

        for (int i = 0; i < count; i++)
        {
            int connections = 0;
            for (int j = i + 1; j < count && connections < maxConnections; j++)
            {
                float dist = Vector3.Distance(particleArray[i].position, particleArray[j].position);
                if (dist < connectDistance)
                {
                    var lineObj = new GameObject("Line");
                    var lr = lineObj.AddComponent<LineRenderer>();
                    lr.material = lineMaterial;
                    lr.startWidth = 0.005f;
                    lr.endWidth = 0.005f;
                    lr.positionCount = 2;
                    lr.SetPosition(0, particleArray[i].position);
                    lr.SetPosition(1, particleArray[j].position);
                    lr.startColor = lr.endColor = new Color(0.5f, 0.8f, 1f, 0.15f);
                    lines.Add(lr);
                    connections++;
                }
            }
        }
    }
}
