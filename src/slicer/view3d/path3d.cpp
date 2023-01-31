#include "path3d.h"
#include <QDebug>
#include <QMessageBox>

Path3d::Path3d()
{
	m_name="";
}

Path3d::Path3d(QString name)
{
	m_name=name;
}

Path3d::~Path3d()
{

}

void Path3d::ClearAllPoints()
{
	m_pathCamera.clear();
}

void Path3d::AddPoints(QVector3D p , QVector3D o,int d)
{
	m_pathCamera.append(new InfosPath(p,o,QVector3D(0,-1,0), d));
}

void Path3d::AddPoints(QVector3D p , QVector3D o,QVector3D u,int d)
{
	m_pathCamera.append(new InfosPath(p,o,u,d));
}

void Path3d::RemovePoint(int index)
{
	if(index>= 0)
		m_pathCamera.removeAt(index);
}

void Path3d::setPosition(int index , QVector3D pos)
{
	m_pathCamera[index]->m_position = pos;
}
void Path3d::setViewCenter(int index , QVector3D target)
{
	m_pathCamera[index]->m_target =target;
}

QVector3D Path3d::getPosition(int index)
{
	return m_pathCamera[index]->m_position;

}
QVector3D Path3d::getViewCenter(int index)
{
	return m_pathCamera[index]->m_target;
}

QVector3D Path3d::getUp(int index)
{
	return m_pathCamera[index]->m_up;
}

int Path3d::getDuration(int index)
{
	return m_pathCamera[index]->m_duration;
}

void Path3d::setDuration(int index, int duree)
{
	m_pathCamera[index]->m_duration = duree;
}

void Path3d::deletePath(QString filepath)
{
	QFile file(filepath+".txt");
	file.remove();
}

void Path3d::SavePath(QString filepath, QString name)
{
	// /data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/ImportExport/IJK/GraphicLayers

	QString directory="3DPath/";
	QDir dir(filepath);
	bool res = dir.mkpath("3DPath");

	QFile file(filepath+directory+name+".txt");
	if(!file.open(QIODevice::WriteOnly | QIODevice::Text))
	{
		QMessageBox msgBox;
		msgBox.setText("Error save path");
		msgBox.setInformativeText("Error :"+file.errorString());
		int ret = msgBox.exec();

		return;
	}
	//qDebug()<<"chemin :"<<m_pathFiles<<directory<<"in2.txt";
	//qDebug()<<"nb element :"<<m_pathCamera.length();
	QTextStream out(&file);
	for (int i=0;i<m_pathCamera.length();i++)
		out<<m_pathCamera[i]->m_position.x()<<"|"<<m_pathCamera[i]->m_position.y()<<"|"<<m_pathCamera[i]->m_position.z()<<"|"<<m_pathCamera[i]->m_target.x()<<"|"<<m_pathCamera[i]->m_target.y()<<"|"<<m_pathCamera[i]->m_target.z()<<"|"<<m_pathCamera[i]->m_duration<<"\n";

}


void Path3d::LoadPath(QString path)
{
	QFile file(path);
	if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		qDebug()<<"path3D Loadpath ouverture du fichier impossible :"<<path;
		return;
	}
	ClearAllPoints();

	QTextStream in(&file);
	QString line = in.readLine();
	while (!line.isNull()) {
		//process_line(line);
		QStringList linesplit = line.split("|");
		if(linesplit.length() >6)
		{
			QVector3D pos(linesplit[0].toFloat(),linesplit[1].toFloat(),linesplit[2].toFloat());
			QVector3D targ(linesplit[3].toFloat(),linesplit[4].toFloat(),linesplit[5].toFloat());
			int dur = linesplit[6].toInt();
			AddPoints(pos,targ,dur);


		}
		line = in.readLine();
	}

}

void Path3d::LoadPath2D(CameraController* cameraCtrl, QMatrix4x4 transfo, QString path)
{
	QFile file(path);
	if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		qDebug()<<"path2D Loadpath ouverture du fichier impossible :"<<path;
		return;
	}
	ClearAllPoints();

	QList<QVector3D> listepoints;
	QTextStream in(&file);
	QString line = in.readLine();
	while (!line.isNull()) {
		//process_line(line);
		QStringList linesplit = line.split("|");
		if(linesplit.length() >1)
		{
			QVector3D pos(linesplit[0].toFloat(),-5000,linesplit[1].toFloat());
			listepoints.append(pos);
			//QVector3D targ(linesplit[3].toFloat(),linesplit[4].toFloat(),linesplit[5].toFloat());
			//int dur = 200;
			//AddPoints(pos,targ,dur);


		}
		line = in.readLine();
	}

	QVector3D lastDir;
	for(int i=0;i<listepoints.count()-1;i++)
	{
		QVector3D pos = transfo * listepoints[i];

		QVector3D pos2 = transfo * listepoints[i+1];

		QVector3D posxz(pos.x(),0.0f,pos.z());
		QVector3D pos2xz(pos2.x(),0.0f,pos2.z());

		QVector3D dir = pos2xz - posxz;
		dir = dir.normalized();

	//	qDebug()<<i<<" position : "<<pos;
		dir.setY(0.7f);

		lastDir = dir;
		//float distance =cameraCtrl->computeHeight(pos);
		//qDebug()<<"Distance :"<<distance;
		//pos.setY(pos.y()-distance *0.01f);

	//	float distance2 =cameraCtrl->computeHeight(pos2);
	//	pos2.setY(pos2.y()-distance2 *0.1f);

		QVector3D targ = pos + dir*100.0f;
		AddPoints( pos, targ,2000);
	}

	QVector3D pos = transfo * listepoints[listepoints.count()-1];
	QVector3D targ = pos + lastDir*100.0f;
	AddPoints( pos, targ,2000);

}


//============================================================================
WidgetPath3d::WidgetPath3d(CameraController* ctrl,QMatrix4x4 sceneTr,Qt3DRender::QCamera* camera,float scaleG, QWidget* parent):QDialog(parent)
{
	m_camera = camera;
	m_scaleGlobal = scaleG;
	m_pathsManager = new Path3dmanager(ctrl,sceneTr);

	//grabKeyboard();


	setWindowTitle("Animator camera");
	setMinimumWidth(500);
	setMinimumHeight(300);
	setModal(false);

	m_nbPoints = 0;

	QLabel* label1= new QLabel(" <b>Name</b>",parent);
	m_editName= new QLineEdit("Trajectoire 0",parent);


	QLabel* label2= new QLabel(" <b>Paths</b>",parent);
	m_comboNamePath= new QComboBox(parent);

	int iconSize=32;
	QHBoxLayout* lay1 = new QHBoxLayout;
	QWidget* widget= new QWidget();
	widget->setMinimumHeight(iconSize);
	QToolButton *newButton = new QToolButton();
	newButton->setIcon(QIcon(":/slicer/icons/path-add.png"));
	newButton->setIconSize(QSize(iconSize,iconSize));
	newButton->setMinimumHeight(iconSize);
	newButton->setMinimumWidth(iconSize);
	//newButton->setIcon(style()->standardPixmap( QStyle::QStyle::SP_FileDialogNewFolder));// QStyle::SC_TitleBarContextHelpButton));
	newButton->setToolTip("New 3D path (touch V to add point)");

	m_playButton = new QToolButton();
	m_playButton->setIcon(QIcon(":/slicer/icons/path-play.png"));
	m_playButton->setIconSize(QSize(iconSize,iconSize));
	//m_playButton->setIcon(style()->standardPixmap( QStyle::SP_MediaPlay));// QStyle::SC_TitleBarContextHelpButton));
	m_playButton->setToolTip("3D path play");
	m_playButton->setCheckable(true);

	QToolButton* saveButton = new QToolButton();
	saveButton->setIcon(QIcon(":/slicer/icons/path-save.png"));
	saveButton->setIconSize(QSize(iconSize,iconSize));
	//saveButton->setIcon(style()->standardPixmap( QStyle::SP_DialogSaveButton));// QStyle::SC_TitleBarContextHelpButton));
	saveButton->setToolTip("3D path save");

	QToolButton* supprButton = new QToolButton();
	supprButton->setIcon(QIcon(":/slicer/icons/path-delete.png"));
	supprButton->setIconSize(QSize(iconSize,iconSize));
	//supprButton->setIcon(style()->standardPixmap( QStyle::SP_MessageBoxCritical));// QStyle::SC_TitleBarContextHelpButton));
	supprButton->setToolTip("3D path delete");

	QHBoxLayout* lay2 = new QHBoxLayout;
	QWidget* widget2= new QWidget();
	//QLabel* labelSpeed= new QLabel(" Speed",parent);
	m_sliderSpeed = new QSlider(Qt::Horizontal);
	m_sliderSpeed->setMinimum(1);
	m_sliderSpeed->setMaximum(m_maxSpeed);
	m_sliderSpeed->setValue(100);
	m_sliderSpeed->setTickInterval(10);
	m_sliderSpeed->setTickPosition(QSlider::TicksBelow);
	m_sliderSpeed->setToolTip("Speed cam");


	m_sliderAltitude = new QSlider(Qt::Horizontal);
	m_sliderAltitude->setMinimum(1);
	m_sliderAltitude->setMaximum(m_altMax);//m_maxAltitude
	m_sliderAltitude->setValue(m_altMax);
	m_sliderAltitude->setTickInterval(10);
	m_sliderAltitude->setTickPosition(QSlider::TicksBelow);
	m_sliderAltitude->setToolTip("Altitude");


	QCheckBox* loopCheck = new QCheckBox("Closed");
	connect(loopCheck,SIGNAL(stateChanged(int)),this,SLOT(setLoop(int)));


	lay1->addWidget(newButton);
	lay1->addWidget(m_playButton);
	lay1->addWidget(saveButton);
	lay1->addWidget(supprButton);


	widget->setLayout(lay1);

	//lay2->addWidget(labelSpeed);
	lay2->addWidget(m_sliderSpeed);
	lay2->addWidget(m_sliderAltitude);
	lay2->addWidget(loopCheck);
	widget2->setLayout(lay2);

	const QStringList headers({tr("Position"), tr("Target"), tr("Duration")});
	listePoints = new QTreeWidget(parent);


	listePoints->setHeaderLabels(headers);
	listePoints->setColumnCount(3);
	listePoints->resizeColumnToContents(0);

	listePoints->setColumnWidth(0,200);
	listePoints->setColumnWidth(1,200);
	listePoints->setColumnWidth(2,60);

	 QLabel* infos = new QLabel("touch V to add point",parent);


	QGridLayout *layout = new QGridLayout();
	layout->addWidget(label1, 0, 0, 1, 1);
	layout->addWidget(m_editName, 0, 1, 1, 1);

	layout->addWidget(label2, 1, 0, 1, 1);
	layout->addWidget(m_comboNamePath, 1, 1, 1, 1);
	layout->addWidget(widget, 2, 0, 1, 1);
	layout->addWidget(widget2, 2, 1, 1, 1);
	layout->addWidget(infos, 3, 0, 1, 1);
	//layout->addWidget(newButton, 2, 0, 1, 1);
	//layout->addWidget(m_playButton, 2, 1, 1, 1);
	//layout->addWidget(saveButton, 2, 2, 1, 1);
	layout->addWidget(listePoints, 4, 0, 4, 3);
	//layout->addWidget(infos, 4, 0, 4, 1);

	 setLayout(layout);

	 connect(newButton, &QPushButton::clicked, this, &WidgetPath3d::newPath3d);
	 connect(m_playButton, &QPushButton::clicked, this, &WidgetPath3d::playPath3d);
	 connect(saveButton, &QPushButton::clicked, this, &WidgetPath3d::savePath3d);
	 connect(supprButton, &QPushButton::clicked, this, &WidgetPath3d::deletePath3d);
	 connect(m_comboNamePath, SIGNAL(currentIndexChanged(int)), this, SLOT(selectedPath(int)));
	 connect(m_sliderSpeed, SIGNAL(valueChanged(int)), this, SLOT(speedChanged(int)));
	 connect(m_sliderAltitude, SIGNAL(valueChanged(int)), this, SLOT(altitudeChanged(int)));

	 connect(listePoints,SIGNAL(itemChanged(QTreeWidgetItem*,int)),this,SLOT(itemEditer(QTreeWidgetItem*,int)));

	 connect(listePoints,SIGNAL(itemClicked(QTreeWidgetItem*,int)),this,SLOT(selectItem(QTreeWidgetItem*,int)));

	 connect(&m_systemWatcher,SIGNAL(fileChanged(QString)),this, SLOT(onFileChanged(QString)));
	 connect(&m_systemWatcher,SIGNAL(directoryChanged(QString)),this, SLOT(onDirChanged(QString)));

	 connect(this, SIGNAL(finished(int)),this,SLOT(closed(int)));

	 /*for(int i=0;i<25;i++)
	 {
		 AddPoints("12.36| 563.2|457.36", QString::number(i));
	 }*/

}

void WidgetPath3d::setLoop(int loop)
{
	if(loop ==0 )
		m_loopPath = false;
	else
		m_loopPath = true;;

}

void WidgetPath3d::selectItem(QTreeWidgetItem* current ,int col)
{
	int index = listePoints->indexOfTopLevelItem(current);

	if(m_lastIndex == index)
	{
		current->setSelected(false);
		m_modeEdition =false;
		m_lastIndex = -1;
		index = -1;
		return;
	}

	if(index >= 0)
	{	m_modeEdition =true;
		QVector3D pos = m_pathsManager->getCurrentPath3d()->getPosition(index);
		QVector3D target = m_pathsManager->getCurrentPath3d()->getViewCenter(index);
		m_camera->setPosition(pos);
		m_camera->setViewCenter(target);
	}

	m_lastIndex = index;

}

void WidgetPath3d::closed(int i)
{
	m_modeEdition =false;
	if(m_animationGrp != nullptr)
	{
		if( m_animationGrp->state() == QAbstractAnimation::Running)
			{
			m_useScale = false;
			m_animationGrp->stop();
			}
	}
}

void WidgetPath3d::onFileChanged(QString s)
{

}


void WidgetPath3d::onDirChanged(QString s)
{

	refreshAllPaths();
}


int WidgetPath3d::GetNbPoints()
{
	return m_nbPoints;
}

void WidgetPath3d::DeletePoint()
{

	Path3d* path3d = m_pathsManager->getCurrentPath3d();
	if(m_lastIndex >= 0)
	{
		path3d->RemovePoint(m_lastIndex);
		QTreeWidgetItem* item = listePoints->takeTopLevelItem(m_lastIndex);
	}
}

void WidgetPath3d::AddPoints(QVector3D position,QVector3D target, int duree)
{
	Path3d* path3d = m_pathsManager->getCurrentPath3d();

	//qDebug()<<" addpoints : "<<m_lastIndex;
	if(m_lastIndex >= 0)
	{
		path3d->setPosition(m_lastIndex, position);
		path3d->setViewCenter(m_lastIndex, target);

		QString positionStr = QString::number((int)position.x())+"|"+QString::number((int)position.y())+"|"+QString::number((int)position.z());
		QString targetStr = QString::number((int)target.x())+"|"+QString::number((int)target.y())+"|"+QString::number((int)target.z());

		QTreeWidgetItem* item = listePoints->takeTopLevelItem(m_lastIndex);
		item->setText(0,positionStr);
		item->setText(1,targetStr);

		listePoints->insertTopLevelItem(m_lastIndex,item);

		listePoints->setCurrentItem(item);

	}
	else
	{
		path3d->AddPoints(position,target,duree);

		QString positionStr = QString::number((int)position.x())+"|"+QString::number((int)position.y())+"|"+QString::number((int)position.z());
		QString targetStr = QString::number((int)target.x())+"|"+QString::number((int)target.y())+"|"+QString::number((int)target.z());
		QString durationStr = QString::number(duree);

		QStringList infos({positionStr,targetStr, durationStr});
		QTreeWidgetItem* child= new QTreeWidgetItem(infos);
		child->setFlags(child->flags() | Qt::ItemIsEditable);


		listePoints->addTopLevelItem(child);
		m_nbPoints++;
	}
}

void WidgetPath3d::speedChanged(int value)
{
	m_coefSpeed = value/200.0f;

	if(m_animationGrp!=nullptr)
	{
		if( m_animationGrp->state() == QAbstractAnimation::Running)
		{
			m_animationGrp->pause();
			Path3d* paths = m_pathsManager->getCurrentPath3d();

			for(int i=0;i<m_animationGrp->animationCount();i++)
			{
				QParallelAnimationGroup *group  = static_cast<QParallelAnimationGroup*>(m_animationGrp->animationAt(i));

				(static_cast<QPropertyAnimation*>(group->animationAt(0)))->setDuration(paths->getDuration(i) / m_coefSpeed);
				(static_cast<QPropertyAnimation*>(group->animationAt(1)))->setDuration(paths->getDuration(i) / m_coefSpeed);

			}

			m_animationGrp->resume();
		}
	}
}

void WidgetPath3d::altitudeChanged(int value)
{
	int maxi  = m_sliderAltitude->maximum();
	m_coefAlt =(float)(m_altMax- value)/m_altMax;
}

void WidgetPath3d::newPath3d()
{
	m_lastIndex = -1;
	m_modeEdition =true;
	m_pathsManager->currentTxt=true;
	QString namepath = m_editName->text();//+"_"+QString::number(m_zScale);
	for (int i = listePoints->topLevelItemCount(); i >= 0; i--)
	{
		listePoints->takeTopLevelItem(i);
	}
	m_nbPoints=0;

	m_pathsManager->AddNewPath(new Path3d(namepath));
	 m_pathsManager->setCurrent(m_pathsManager->getNbPath()-1);
	 AddPoints(m_camera->position(),m_camera->viewCenter(),2000);


}

void WidgetPath3d::playPath3d()
{

	if( m_pathsManager->currentTxt==false)
	{

		playPath3d2dp();
	}
	else
	{
		playPath3dTxt();
	}

}

void WidgetPath3d::playPath3d2dp()
{

	if(m_animationGrp != nullptr )
	{
		if( m_animationGrp->state() == QAbstractAnimation::Running)
		{
			m_useScale = false;
			m_animationGrp->stop();
			return;
		}

		delete m_animationGrp;
		m_animationGrp = nullptr;
	}
	m_animationGrp = new QSequentialAnimationGroup;
	connect(m_animationGrp,SIGNAL(finished()),this,SLOT(onFinish()));

	Path3d* paths = m_pathsManager->getCurrentPath3d();

	if(paths ==nullptr) return;

	QVector3D posDepart= paths->getPosition(0)*m_scaleGlobal;
	QVector3D targetDepart=paths->getViewCenter(0)*m_scaleGlobal;

	float scaleZ = 1.0f;//m_zScale*0.01f;

	float distance =scaleZ * m_pathsManager->m_cameraCtrl->computeHeight(posDepart);
	float lastdistance =m_zScale*0.01f * m_pathsManager->m_cameraCtrl->computeHeight(posDepart);
	float valeur = -m_coefAlt;

	float ajoutY = 0.0f;
	if( distance <0 )ajoutY  = 2.0* lastdistance ;

	posDepart.setY(ajoutY+ posDepart.y()-distance *valeur);
	targetDepart.setY(ajoutY +targetDepart.y() -distance *valeur);

	for(int i=1;i< paths->getLength();i++)
	{
		QVector3D pos =  paths->getPosition(i)*m_scaleGlobal;
		QVector3D targ =  paths->getViewCenter(i)*m_scaleGlobal;


		float distance =scaleZ *m_pathsManager->m_cameraCtrl->computeHeight(pos);

		pos.setY(ajoutY+ pos.y()-distance *valeur);
		targ.setY(ajoutY+targ.y() -distance *valeur);

		int duree =paths->getDuration(i)/m_coefSpeed;

		QPropertyAnimation* anim1= new QPropertyAnimation(m_camera,"viewCenter");
		anim1->setDuration(duree);
		anim1->setStartValue(targetDepart);
		anim1->setEndValue(targ);


		QPropertyAnimation* anim2 = new QPropertyAnimation(m_camera,"position");
		anim2->setDuration(duree);
		anim2->setStartValue(posDepart);
		anim2->setEndValue(pos);


		posDepart= pos;//paths->getPosition(i);
		targetDepart=targ;// paths->getViewCenter(i);


		 QParallelAnimationGroup *group = new QParallelAnimationGroup;
		 group->addAnimation(anim1);
		 group->addAnimation(anim2);


		 m_animationGrp->addAnimation(group);

	}

	if(m_loopPath)
	{
		QVector3D pos =  paths->getPosition(0)*m_scaleGlobal;
			QVector3D targ =  paths->getViewCenter(0)*m_scaleGlobal;


			float distance =scaleZ *m_pathsManager->m_cameraCtrl->computeHeight(pos);

			pos.setY(ajoutY+ pos.y()-distance *valeur);
			targ.setY(ajoutY+targ.y() -distance *valeur);

			int duree =paths->getDuration(0)/m_coefSpeed;

			QPropertyAnimation* anim1= new QPropertyAnimation(m_camera,"viewCenter");
			anim1->setDuration(duree);
			anim1->setStartValue(targetDepart);
			anim1->setEndValue(targ);


			QPropertyAnimation* anim2 = new QPropertyAnimation(m_camera,"position");
			anim2->setDuration(duree);
			anim2->setStartValue(posDepart);
			anim2->setEndValue(pos);

			QParallelAnimationGroup *group = new QParallelAnimationGroup;
			group->addAnimation(anim1);
			group->addAnimation(anim2);


			m_animationGrp->addAnimation(group);

	}

	m_camera->setUpVector(QVector3D (0,-1,0));

	m_useScale = true;
	m_animationGrp->start();
}

void WidgetPath3d::onFinish()
{
	m_useScale = false;
}

void WidgetPath3d::playPath3dTxt()
{

	if(m_animationGrp != nullptr )
	{
		if( m_animationGrp->state() == QAbstractAnimation::Running)
		{
			m_useScale = false;
			m_animationGrp->stop();
			return;
		}

		delete m_animationGrp;
		m_animationGrp = nullptr;
	}
	m_animationGrp = new QSequentialAnimationGroup;

	Path3d* paths = m_pathsManager->getCurrentPath3d();


	if(paths ==nullptr) return;

	float coef = 1.0f;//m_scaleGlobal;

	QVector3D posDepart= paths->getPosition(0)*coef;
	QVector3D targetDepart=paths->getViewCenter(0)*coef;

	float distance =m_pathsManager->m_cameraCtrl->computeHeight(posDepart);

	//float valeur = -m_coefAlt;

	//posDepart.setY(posDepart.y()-distance *valeur);
	//targetDepart.setY(targetDepart.y() -distance *valeur);


	for(int i=1;i< paths->getLength();i++)
	{
		QVector3D pos =  paths->getPosition(i)*coef;
	//	QVector3D pos2 =  paths->getPosition(i+1);
		QVector3D targ =  paths->getViewCenter(i)*coef;


		//	float distance =m_pathsManager->m_cameraCtrl->computeHeight(pos);
		//	pos.setY(pos.y()-distance *valeur);
		//	targ.setY(targ.y() -distance *valeur);


		int duree =paths->getDuration(i)/m_coefSpeed;


		QPropertyAnimation* anim1= new QPropertyAnimation(m_camera,"viewCenter");
		anim1->setDuration(duree);
		anim1->setStartValue(targetDepart);
		anim1->setEndValue(targ);
		//animation->start();

		QPropertyAnimation* anim2 = new QPropertyAnimation(m_camera,"position");
		anim2->setDuration(duree);
		anim2->setStartValue(posDepart);
		anim2->setEndValue(pos);
	///	animation2->start();

		posDepart= pos;//paths->getPosition(i);
		targetDepart=targ;// paths->getViewCenter(i);


		 QParallelAnimationGroup *group = new QParallelAnimationGroup;
		 group->addAnimation(anim1);
		 group->addAnimation(anim2);


		 m_animationGrp->addAnimation(group);

	}
	if(m_loopPath)
	{
		QVector3D pos =  paths->getPosition(0)*m_scaleGlobal;
		QVector3D targ =  paths->getViewCenter(0)*m_scaleGlobal;

		int duree =paths->getDuration(0)/m_coefSpeed;

		QPropertyAnimation* anim1= new QPropertyAnimation(m_camera,"viewCenter");
		anim1->setDuration(duree);
		anim1->setStartValue(targetDepart);
		anim1->setEndValue(targ);


		QPropertyAnimation* anim2 = new QPropertyAnimation(m_camera,"position");
		anim2->setDuration(duree);
		anim2->setStartValue(posDepart);
		anim2->setEndValue(pos);

		QParallelAnimationGroup *group = new QParallelAnimationGroup;
		group->addAnimation(anim1);
		group->addAnimation(anim2);


		m_animationGrp->addAnimation(group);

	}



	m_camera->setUpVector(QVector3D (0,-1,0));
	m_useScale = false;
	m_animationGrp->start();
}

void WidgetPath3d::deletePath3d()
{
	if(m_animationGrp!=nullptr)
	{
		if( m_animationGrp->state() == QAbstractAnimation::Running)
		{
			m_useScale = false;
			m_animationGrp->stop();
		}
	}

	Path3d* paths = m_pathsManager->getCurrentPath3d();

	if(paths == nullptr) return ;
	paths->deletePath(m_comboNamePath->currentText());



	int index = m_comboNamePath->currentIndex();

	m_pathsManager->SupprPath(index);
	QString nametext = m_comboNamePath->itemText(index);
	QString fullpath =m_pathsManager->getPathFiles()+ "3DPath/"+nametext+".2dp";
	QString fullpath2 =m_pathsManager->getPathFiles()+ "3DPath/"+nametext+".txt";
	QFile::remove(fullpath);
	QFile::remove(fullpath2);

	//qDebug()<<" remove path "<<fullpath;


	for (int i = listePoints->topLevelItemCount(); i >= 0; i--)
	{
		listePoints->takeTopLevelItem(i);
	}
	m_nbPoints=0;

	m_comboNamePath->removeItem(index);





}

void WidgetPath3d::savePath3d()
{
	m_modeEdition =false;
	Path3d* paths = m_pathsManager->getCurrentPath3d();

	QString temp = m_editName->text();
	QStringList resplit = temp.split("#");
	QString namepath = m_editName->text()+"#"+QString::number(m_zScale);
	if(resplit.length()>1)
	{
		namepath = resplit[0]+"#"+QString::number(m_zScale);
	}



	int indexfind = m_comboNamePath->findText(namepath);
	/*if(indexfind < 0)
	{
		m_comboNamePath->addItem(namepath);
	}*/


	paths->SavePath(m_pathsManager->getPathFiles(),namepath);
}

void WidgetPath3d::itemEditer(QTreeWidgetItem* item,int c)
{

	Path3d* paths = m_pathsManager->getCurrentPath3d();
	int index = listePoints->indexOfTopLevelItem(item);
	int value = item->text(c).toInt();
	paths->setDuration(index,value );

}

void WidgetPath3d::selectedPath (int index)
{
//	qDebug()<<" selected path : "<<index;

	m_modeEdition =false;
	if(index < 0) return;
	m_pathsManager->setCurrent(index);
	Path3d* paths = m_pathsManager->getCurrentPath3d();

	for (int i = listePoints->topLevelItemCount(); i >= 0; i--)
	{
		listePoints->takeTopLevelItem(i);
	}


	m_nbPoints=0;
	QString nametext = m_comboNamePath->itemText(index);
	QString fullpath =m_pathsManager->getPathFiles()+ "3DPath/"+nametext+".txt";

	QFile file(fullpath);
	if(!file.exists())
	{
		fullpath = fullpath.replace(".txt",".2dp");

		paths->LoadPath2D(m_pathsManager->m_cameraCtrl, m_pathsManager->getTransfo(), fullpath);
		m_pathsManager->currentTxt= false;

	}
	else
	{
		paths->LoadPath(fullpath);
		m_pathsManager->currentTxt=true;
	}

	m_editName->setText(nametext);
	//qDebug()<<" nb points :"<<paths->getLength();
	for(int i=0;i<paths->getLength();i++)
	{
		QVector3D position = paths->getPosition(i);
		QVector3D target = paths->getViewCenter(i);
		int duree = paths->getDuration(i);
		QString positionStr = QString::number((int)position.x())+"|"+QString::number((int)position.y())+"|"+QString::number((int)position.z());
		QString targetStr = QString::number((int)target.x())+"|"+QString::number((int)target.y())+"|"+QString::number((int)target.z());
		QString durationStr = QString::number(duree);

		QStringList infos({positionStr,targetStr, durationStr});
		QTreeWidgetItem* child= new QTreeWidgetItem(infos);
		child->setFlags(child->flags() | Qt::ItemIsEditable);

		listePoints->addTopLevelItem(child);
		m_nbPoints++;
	}

}

void WidgetPath3d::refreshAllPaths()
{

	QDir dir(m_pathsManager->getPathFiles()+"3DPath/");

	QFileInfoList fileinfo = dir.entryInfoList();

	for (int i = 0; i < fileinfo.size(); ++i)
	{
		 QFileInfo fileInfo = fileinfo.at(i);
		 if(fileInfo.suffix() =="txt" || fileInfo.suffix()=="2dp")
		 {

			 if(!m_pathsManager->contains(fileInfo.baseName()))
			{

				 QString namepath = fileInfo.baseName();//+"_"+QString::number(m_zScale);
				// qDebug()<<"namepath ==>"<<namepath;
				 m_pathsManager->AddNewPath(new Path3d(namepath));

				 m_comboNamePath->addItem(namepath);
				 m_comboNamePath->setCurrentIndex(m_comboNamePath->count()-1);
			}
		 }
	}
}


void WidgetPath3d::speedAnimChanged(int v)
{
	m_maxSpeed= v;
	if(m_sliderSpeed )  m_sliderSpeed->setMaximum(v);
}

void WidgetPath3d::altitudeAnimChanged(int v)
{
	m_altMax =v;
	if(m_sliderAltitude) m_sliderAltitude->setMaximum(v);
}



































