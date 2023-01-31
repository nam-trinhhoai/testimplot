#ifndef PATH3D_H
#define PATH3D_H
#include <QVector3D>
#include <Qt3DRender/QCamera>
#include <QDialog>
#include <QLabel>
#include <QLineEdit>
#include <QComboBox>
#include <QPushButton>
#include <QGroupBox>
#include <QTreeWidget>
#include <QToolButton>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QCheckBox>
#include <QFile>
#include <QDir>
#include <QPropertyAnimation>
#include <QSequentialAnimationGroup>
#include <QParallelAnimationGroup>
#include "cameracontroller.h"
#include <QFileSystemWatcher>
#include <QKeyEvent>

#include <boost/filesystem.hpp>

class CameraController;

class InfosPath
{
public:
	QVector3D m_position;
	QVector3D m_target;
	QVector3D m_up;
	int m_duration;

	InfosPath(){}
	InfosPath( QVector3D p, QVector3D t,QVector3D u, int d)
	{
		m_position = p;
		m_target = t;
		m_up = u;
		m_duration = d;
	}
	~InfosPath(){}


};

class Path3d
{

public:
	Path3d();
	Path3d(QString name);

	~Path3d();

	void ClearAllPoints();
	void AddPoints(QVector3D p , QVector3D o, int d);
	void AddPoints(QVector3D p , QVector3D o, QVector3D u, int d);
	void SavePath(QString filepath, QString name);

	void RemovePoint(int index);

	void deletePath(QString);

	void LoadPath(QString path);
	void LoadPath2D(CameraController* cameraCtrl,QMatrix4x4 transfo,QString path);

	int getLength()
	{
		return m_pathCamera.length();
	}


	void setPosition(int index , QVector3D);
	void setViewCenter(int index , QVector3D);

	QVector3D getPosition(int index);
	QVector3D getViewCenter(int index);

	QVector3D getUp(int index);
	int getDuration(int index);

	void setDuration(int index, int duree);


	QString m_name="";
private:
	QList<InfosPath*> m_pathCamera;

};


class Path3dmanager
{
public:
	Path3dmanager()
	{

	}

	Path3dmanager(CameraController* ctrl, QMatrix4x4 sceneTr)
	{

		m_cameraCtrl =ctrl;
		m_sceneTransform = sceneTr;
	}
	void AddNewPath(Path3d* p)
	{
		m_pathsList.append(p);
	}

	void SupprPath(int i)
	{
		getPath3d( i)->deletePath("");
		m_pathsList.removeAt(i);
	}

	Path3d* getPath3d( int index)
	{
		return m_pathsList[index];
	}

	Path3d* getCurrentPath3d( )
	{
		//qDebug()<<" m_indexCurrent "<<m_indexCurrent<<" , count :"<<m_pathsList.count();
		if(m_indexCurrent>=0 && m_pathsList.count() > 0)
			return m_pathsList[m_indexCurrent];
		else
			return nullptr;
	}


	void setPathFiles(QString s)
	{
		m_pathFiles = s;
	}

	QString getPathFiles()
	{
		return m_pathFiles;
	}

	bool contains(QString s)
	{
		//qDebug()<<" liste count : "<<m_pathsList.count();
		for(int i=0;i<m_pathsList.count();i++)
		{
			//qDebug()<<" contains : "<<m_pathsList[i]->m_name<<" != " <<s;
			if( m_pathsList[i]->m_name == s )
			{
				return true;
			}
		}
		return false;
	}

	void LoadPaths()
	{

		QDir dir(m_pathFiles+"3DPath/");

		QFileInfoList fileinfo = dir.entryInfoList();

		for (int i = 0; i < fileinfo.size(); ++i)
		{
			 QFileInfo fileInfo = fileinfo.at(i);
			 if(fileInfo.suffix() =="txt" || fileInfo.suffix()=="2dp")
			 {
				 //qDebug()<<"==>"<<fileInfo.fileName();

				 m_pathsList.append(new Path3d(fileInfo.baseName()));
				 //LoadPath(fileInfo.filePath() );
			 }
		}

	}

	void setCurrent (int i)
	{
		m_indexCurrent = i;
	}
	int getCurrent()
	{
		return m_indexCurrent;
	}

	int getNbPath()
	{
		return m_pathsList.size();
	}

	QMatrix4x4 getTransfo()
	{
		return m_sceneTransform;
	}

	void setTransfo(QMatrix4x4 m)
	{
		m_sceneTransform = m;
	}




	CameraController* m_cameraCtrl;
	bool currentTxt = false;
private :
	QList<Path3d*> m_pathsList;
	QString m_pathFiles="";
	int m_indexCurrent = -1;
	QMatrix4x4 m_sceneTransform;





};

class WidgetPath3d: public QDialog
{
	 Q_OBJECT
private:

	 int m_lastIndex = -1;
	Qt3DRender::QCamera* m_camera;
	QTreeWidget* listePoints;
	int m_nbPoints;

	QLineEdit* m_editName;
	QComboBox* m_comboNamePath;
	QToolButton* m_playButton;
	QSlider* m_sliderAltitude= nullptr;
	QSlider* m_sliderSpeed=nullptr;

	Path3dmanager* m_pathsManager= nullptr;
	QSequentialAnimationGroup *m_animationGrp = nullptr;


	int m_zScale;

	QFileSystemWatcher m_systemWatcher;

	float m_coefSpeed = 1.0f;
	float m_coefAlt = 0.0f;

	int m_maxSpeed=200;
	int m_maxAltitude = 400;
	float m_scaleGlobal = 1.0f;
	float m_altMax = 500.0f;


public slots:
	void newPath3d();
	void playPath3d();

	void playPath3dTxt();
	void playPath3d2dp();
	void savePath3d();
	void deletePath3d();
	void itemEditer(QTreeWidgetItem*,int);
	void selectItem(QTreeWidgetItem* current ,int);
	void selectedPath(int index);
	void speedChanged(int);
	void altitudeChanged(int);
	void onFileChanged(QString);
	void onDirChanged(QString);
	void closed(int);

	void onFinish();
	void speedAnimChanged(int);
	void altitudeAnimChanged(int);

	void setLoop(int);

public:
	bool m_modeEdition =false;
	bool m_useScale = false;
	bool m_loopPath=false;

	WidgetPath3d(CameraController* ctrl,QMatrix4x4 sceneTr, Qt3DRender::QCamera* camera,float scaleG, QWidget* parent);

	int GetNbPoints();

	bool isRunning()
	{
		if(m_animationGrp== nullptr) return false;
		if(m_animationGrp->state() == QAbstractAnimation::Running )
			return true;
		return false;
	}

	void AddPoints(QVector3D position,QVector3D target, int duree);

	void DeletePoint();

	void setZScale(int z)
		{
			m_zScale= z;
		}

	void setPathFiles(QString path)
	{
		QString path3d = path+"3DPath/";


		// create dir
		boost::filesystem::path searchPath(path3d.toStdString());
		bool dirExists = boost::filesystem::exists(searchPath);
		bool valid = true;
		QStringList dirsToCreate;
		while (!dirExists && valid) {
			dirsToCreate.insert(0, QString(searchPath.filename().c_str()));
			valid = searchPath.has_parent_path();
			if (valid) {
				searchPath = searchPath.parent_path();
				dirExists = boost::filesystem::exists(searchPath);
			}
		}
		if (dirExists && valid && dirsToCreate.count()>0) {
			QDir searchDir(QString(searchPath.c_str()));
			valid = searchDir.mkpath(dirsToCreate.join(QDir::separator()));
		}

		m_systemWatcher.removePath(m_pathsManager->getPathFiles()+"3DPath/");
		m_pathsManager->setPathFiles(path);


		m_systemWatcher.addPath(path+"3DPath/");

		//qDebug()<<"==> set path "<<m_systemWatcher.directories();

	}
	void setSceneTransform(QMatrix4x4 m)
	{
		m_pathsManager->setTransfo( m);
	}

	void newPath()
	{
		m_pathsManager->LoadPaths();
		for(int i=0;i<m_pathsManager->getNbPath();i++)
		{
			QString nom = m_pathsManager->getPath3d(i)->m_name;
			m_comboNamePath->addItem(nom);
		}

		/*if( m_pathsManager->getNbPath() ==0)
		{
			m_pathsManager->AddNewPath(new Path3d());

		}*/
		 if(m_pathsManager->getNbPath()>0) m_pathsManager->setCurrent(0);
	}

	void refreshAllPaths();

	void keyPressEvent(QKeyEvent * e)
	{
	//	qDebug()<<" patch 3d :keyPressEvent";

		if(e->key() == Qt::Key_Delete)
		{
			DeletePoint();
		}

		if(e->key() == Qt::Key_V)
		{
			AddPoints(m_camera->position(),m_camera->viewCenter(),2000);
		}

	}

};


#endif
