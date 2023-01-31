#include "scenemultimanager.h"
#include <QDebug>

SceneMultiManager::SceneMultiManager(QObject *parent): QObject(parent)
{
	m_maxNbView3D =4;
	m_nbView3D =1;
}

SceneMultiManager::~SceneMultiManager()
{

}


void SceneMultiManager::addView()
{
	qDebug()<<" ajout d'une nouvelle vue!";
	if(m_nbView3D < m_nbView3D)
	{

		m_nbView3D++;
	}

}
