#include "nurbsmanager.h"
#include <QFileInfo>
//#include "affine2dtransformation.h"
#include "randomTransformation.h"
#include "GraphEditor_ListBezierPath.h"
#include "folderdata.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "fixedlayersfromdatasetandcube.h"

NurbsManager::NurbsManager(QObject* parent):QObject(parent)
{

}

NurbsManager::~NurbsManager()
{

}

void NurbsManager::setColorNurbs(QColor col)
{

	if(m_currentNurbs != nullptr) m_currentNurbs->setColorNurbs(col);
	else
		qDebug()<<"m_currentNurbs est NULLLLLLL";

}

void NurbsManager::setColorNurbs(QString nameNurbs, QColor col)
{
	int indice = getIndex(nameNurbs);
	if(indice >=0 )
	{
		if( m_listeNurbs[indice] != nullptr) m_listeNurbs[indice]->setColorNurbs(col);

	}
}

void NurbsManager::setColorDirectrice(QColor col)
{
	if(m_currentNurbs != nullptr) m_currentNurbs->setColorDirectrice(col);

}


void NurbsManager::setPositionLight(QVector3D pos)
{
	for(int i=0;i<m_listeNurbs.count();i++)
	{
		m_listeNurbs[i]->setPositionLight( pos);
	}
}

Manager* NurbsManager::getNurbsFromRandom(RandomView3D* r)
{
	for(int i=0;i<m_listeNurbs.count();i++)
	{
		if( m_listeNurbs[i]->getRandom3d() == r)
			return m_listeNurbs[i];
	}
	return nullptr;
}

void NurbsManager::setCurrentRandom(RandomView3D* rand)
{
	if(m_currentNurbs != nullptr) m_currentNurbs->setRandom3D(rand);
}
RandomView3D* NurbsManager::getCurrentRandom()
{
	if(m_currentNurbs != nullptr)
		return m_currentNurbs->getRandom3d();

	return nullptr;
}

RandomView3D* NurbsManager::getRandom(QString name)
{
	int indice = getIndex(name);
	if(indice >=0 )
	{
		m_listeNurbs[indice]->getRandom3d();
	}

	return nullptr;
}


Manager* NurbsManager::exists(QString name)
{
	for(int i=0;i<m_listeNurbs.count();i++)
	{
		if(m_listeNurbs[i]->getNameId() == name)
			return m_listeNurbs[i];
	}
	return nullptr;
}

void NurbsManager::exportNurbsObj(QString s)
{
	if(m_currentNurbs != nullptr)
		m_currentNurbs->exportNurbsObj(s);
}

int NurbsManager::getLayerTime()
{

	for (IData* idata : m_workingset->folders().horizonsIso->data())
	{
		if (FixedRGBLayersFromDatasetAndCube* layer = dynamic_cast<FixedRGBLayersFromDatasetAndCube*>(idata))
		{
			if(layer != nullptr)
			{
				return layer->getCurrentTime();
			}
		}
		if (FixedLayersFromDatasetAndCube* layer = dynamic_cast<FixedLayersFromDatasetAndCube*>(idata))
		{
			if(layer != nullptr)
			{
				return layer->getCurrentTime();
			}
		}
	}
	return 0;
}

void NurbsManager::createNurbsSimple( QString nameNurbs,IsoSurfaceBuffer buffer,Qt3DCore::QEntity* root,QObject* parent)
{
	int time = getLayerTime();
	QVector3D posCam;
	Manager* nurbs = new Manager(m_workingset, nameNurbs,m_precision,posCam,root,buffer,time, parent);
	m_workingset->addNurbs(nurbs->getNurbsData());
	nurbs->getNurbsData()->setAllDisplayPreference(true);



	m_listeNurbs.push_back(nurbs);

	m_currentNurbs= nurbs;
}

void NurbsManager::addNurbsFromTangent(QVector3D posCam, Qt3DCore::QEntity* root,QObject* parent,QVector<QVector3D> listepoints,IsoSurfaceBuffer surface,QString nameNurbs,GraphEditor_ListBezierPath* path ,QMatrix4x4 transform,QColor col)
{


	Manager* nurbs = exists(nameNurbs);
	if(nurbs ==nullptr)
	{

		int time = getLayerTime();
		nurbs = new Manager(m_workingset, nameNurbs,m_precision,posCam,root,surface,time, parent);
		m_workingset->addNurbs(nurbs->getNurbsData());
		nurbs->getNurbsData()->setAllDisplayPreference(true);



		m_listeNurbs.push_back(nurbs);

		m_currentNurbs= nurbs;

	}


//	Manager* nurbs = new Manager(m_workingset, nameNurbs,m_precision,posCam,root,surface, parent);

//	m_workingset->addNurbs(nurbs->getNurbsData());
//	nurbs->getNurbsData()->setDisplayPreference(true);

	std::vector<QVector3D> listepts;

	for(int i=0;i<listepoints.count()-1;i++)
	{
		listepts.push_back(listepoints[i]);
	}

	path->setNameNurbs(nameNurbs);
	//nurbs->createDirectriceFromTangent(listepts);
	nurbs->createDirectriceFromTangent(path,transform,col);

	nurbs->setDirectriceOk(listepoints.count());

	connect(nurbs,SIGNAL(sendNurbsY(QVector3D,QVector3D)),this, SLOT(receiveNurbsY(QVector3D,QVector3D)));
	connect(nurbs,SIGNAL(sendCurveDataTangent(QVector<PointCtrl>,bool ,QPointF)),this, SLOT(receiveCurveDataTangent(QVector<PointCtrl>,bool,QPointF)));
	connect(nurbs,SIGNAL(sendCurveDataTangentOpt(GraphEditor_ListBezierPath*)),this, SLOT(receiveCurveDataTangentOpt(GraphEditor_ListBezierPath*)));
	connect(nurbs,SIGNAL(sendCurveDataTangent2(QVector<QVector3D>,QVector<QVector3D>,bool ,QPointF)),this, SLOT(receiveCurveDataTangent2(QVector<QVector3D>,QVector<QVector3D>,bool,QPointF)));
	connect(nurbs,SIGNAL(sendAnimationCam(int,QVector3D)),this, SLOT(setAnimationCamera(int, QVector3D)));


}

void NurbsManager::addNurbs(QVector3D posCam, Qt3DCore::QEntity* root,QObject* parent,QVector<QVector3D> listepoints,IsoSurfaceBuffer surface)
{
	Manager* nurbs = new Manager(m_workingset, getUniqueName(),m_precision,posCam,root,surface,getLayerTime(), parent);
	//nurbs->setNameId(getUniqueName());

	m_workingset->addNurbs(nurbs->getNurbsData());
	nurbs->getNurbsData()->setAllDisplayPreference(true);

	for(int i=0;i<listepoints.count();i++)
	{
		nurbs->curveDrawMouseBtnDownGeomIsect(listepoints[i]);
	}
	nurbs->setDirectriceOk(listepoints.count());


	connect(nurbs,SIGNAL(sendNurbsY(QVector3D,QVector3D)),this, SLOT(receiveNurbsY(QVector3D,QVector3D)));
	connect(nurbs,SIGNAL(sendCurveData(std::vector<QVector3D>,bool )),this, SLOT(receiveCurveData(std::vector<QVector3D>,bool)));
	connect(nurbs,SIGNAL(sendAnimationCam(int,QVector3D)),this, SLOT(setAnimationCamera(int, QVector3D)));

	m_listeNurbs.push_back(nurbs);

	m_currentNurbs= nurbs;
}

void NurbsManager::setAnimationCamera(int button, QVector3D pos)
{

	/*if(button==1)*/
	emit sendAnimationCam(button,pos);
	/*if(button==2) emit sendPosition(pos);*/
}

void NurbsManager::receiveCurveData(std::vector<QVector3D> listepoints,bool isopen)
{
	if(isopen == false)
	{
		listepoints.erase(listepoints.end());
	}
	emit sendCurveData(listepoints,isopen);
}

void NurbsManager::receiveCurveDataTangent(QVector<PointCtrl> listepoints,bool isopen,QPointF cross)
{
	/*if(isopen == false)
	{
		listepoints.erase(listepoints.end());
	}*/
	emit sendCurveDataTangent(listepoints,isopen,cross);
}

void NurbsManager::receiveCurveDataTangent2(QVector<QVector3D> listepoints3D,QVector<QVector3D> globalTangente3D,bool isopen,QPointF cross)
{

	emit sendCurveDataTangent2(listepoints3D,globalTangente3D,isopen,cross,m_currentNurbs->getNameId());
}

void NurbsManager::receiveCurveDataTangentOpt(GraphEditor_ListBezierPath* path)
{

	emit sendCurveDataTangentOpt(path);
}

void NurbsManager::receiveNurbsY(QVector3D pos, QVector3D normal)
{
	emit sendNurbsY(pos, normal);
}

void NurbsManager::deleteNurbs(int index)
{

	if(index <m_listeNurbs.count())
	{
		m_workingset->removeNurbs(m_listeNurbs[index]->getNurbsData());
		//delete m_listeNurbs[index];
		m_listeNurbs[index]->deleteLater();
		m_listeNurbs.removeAt(index);
	}
}

void NurbsManager::destroyNurbs(QString name)
{
	int indice = getIndex(name);
	if(indice >=0 )
	{
		m_listeNurbs[indice]->deleteRandom3d();
		m_workingset->removeNurbs(m_listeNurbs[indice]->getNurbsData());
		//delete m_listeNurbs[indice];
		m_listeNurbs[indice]->deleteLater();
		m_listeNurbs.removeAt(indice);
	}
}


void NurbsManager::deleteCurrentNurbs()
{

	if(m_currentNurbs != nullptr)
	{
		m_workingset->removeNurbs(m_currentNurbs->getNurbsData());
		m_listeNurbs.removeOne(m_currentNurbs);

	//m_currentNurbs->deleteRandom3d();
		//delete m_currentNurbs;
		m_currentNurbs->deleteLater();
		m_currentNurbs =nullptr;
	}
	/*else
	{
		qDebug()<<"m_currentNurbs etait null ";
	}*/
}

void NurbsManager::deleteNurbs(QString s)
{

	int indice = getIndex(s);
	if(indice >=0 )
	{
		m_listeNurbs[indice]->deleteRandom3d();
		m_workingset->removeNurbs(m_listeNurbs[indice]->getNurbsData());
		//delete m_listeNurbs[indice];
		m_listeNurbs[indice]->deleteLater();
		m_listeNurbs.removeAt(indice);
	}
}

void NurbsManager:: deleteCurrentGeneratrice(QString s)
{
	int indice = getIndex(s);
	if(indice >=0 )
	{
		m_listeNurbs[indice]->deleteGeneratrice();
	}
}


void NurbsManager::addXSection()
{

}

void NurbsManager::setSliderXsection(float pos)
{
	if(m_currentNurbs != nullptr)
	{
		m_currentNurbs->setSliderXsectPos(pos);
	}
}


float NurbsManager::getSliderXSection()
{
	return m_currentNurbs->getXSection();
}

QVector3D NurbsManager::setSliderXsection(float pos,QVector3D position, QVector3D normal)
{
	if(m_currentNurbs != nullptr)
	{
		return m_currentNurbs->setSliderXsectPos(pos,position,normal);
	}
}




void NurbsManager::createNewXSection(float pos)
{
	if(m_currentNurbs != nullptr)
	{
		m_currentNurbs->addinbetweenXsection(pos);
	}
}

void NurbsManager::createNewXSectionClone(float pos)
{
	if(m_currentNurbs != nullptr)
	{
		m_currentNurbs->addinbetweenXsectionClone(pos);
	}
}

void NurbsManager::updateDirectriceWithTangent(GraphEditor_ListBezierPath* path,QMatrix4x4 transform,QColor col)
{
	qDebug()<<" updateDirectriceWithTangent "<<m_currentNurbs;
	if(m_currentNurbs != nullptr)
	{

		m_currentNurbs->createDirectriceFromTangent(path,transform, col);
	}
	else
	{
		qDebug()<<"Error: m_currentNurbs est null";
	}


/*	if(m_currentNurbs != nullptr)
	{
		std::vector<QVector3D > listePts;
		for(int i=0;i<listepoints.count();i++)
			listePts.push_back(listepoints[i]);

		m_currentNurbs->createDirectriceFromTangent(listePts);
	}
	else
	{
		qDebug()<<"Error: m_currentNurbs est null";
	}*/
}
void NurbsManager::updateDirectrice(QVector<QVector3D> listepoints)
{
	if(m_currentNurbs != nullptr)
	{
		m_currentNurbs->updateDirectriceCurve(listepoints);
		m_currentNurbs->setDirectriceOk(listepoints.count());
	}
	else
	{
		qDebug()<<"Error: m_currentNurbs est null";
	}
}

void NurbsManager::addGeneratriceFromTangent(QVector<QVector3D> listepoints, int index,bool isopen)
{
	qDebug()<<" ==> OBSOLETE";
	/*if(m_currentNurbs != nullptr)
	{
		m_currentNurbs->createGeneratriceFromTangent(listepoints,index,isopen);
		//m_currentNurbs->endCurve();
	}**/
}

void NurbsManager::addGeneratrice(QVector<QVector3D> listepoints, int index,bool isopen)
{
	if(m_currentNurbs != nullptr)
	{
		/*if(m_currentNurbs->m_isTangent)
		{
			m_currentNurbs->createGeneratriceFromTangent(listepoints,index,isopen);
		}
		else
		{*/
			m_currentNurbs->curveDrawSection(listepoints,index,isopen);
			m_currentNurbs->endCurve();
		//}
	}
}


void NurbsManager::addGeneratrice(QVector<PointCtrl> listeCtrls,QVector<QVector3D> listepoints, int index,QVector<QVector3D>  listeCtrl3D,QVector<QVector3D>  listeTangente3D,QVector3D cross3d,bool isopen,QPointF cross)
{
	if(m_currentNurbs != nullptr)
	{
		if(m_currentNurbs->m_isTangent)
		{

			m_currentNurbs->createGeneratriceFromTangent(listeCtrls,listepoints,listeCtrl3D,listeTangente3D,cross3d,index,isopen,cross);
		}

	}
}

void NurbsManager::addGeneratrice(GraphEditor_ListBezierPath* path,RandomTransformation* transfo)
{
	if(m_currentNurbs != nullptr)
	{
		if(m_currentNurbs->m_isTangent)
		{

			m_currentNurbs->createGeneratriceFromTangent(path,transfo);
		}

	}
}

void NurbsManager::updateGeneratrice(QVector<QVector3D> listepoints, int index)
{
	/*if(m_currentNurbs != nullptr)
	{
		m_currentNurbs->curveDrawSection(listepoints,index);
		m_currentNurbs->endCurve();
	}*/
}

void NurbsManager::SelectNurbs(int index)
{

	if(index >=0 && index < m_listeNurbs.count())
	{
		m_currentNurbs = m_listeNurbs[index];
		emit sendColorNurbs(m_currentNurbs->getColorDirectrice() ,m_currentNurbs->getColorNurbs(),m_currentNurbs->getPrecision(),m_currentNurbs->getModeEditable(),m_currentNurbs->getTimerLayer());

	}
	else
		m_currentNurbs = nullptr;
}

void NurbsManager::SelectNurbs(QString s)
{
	int indice = getIndex(s);
	if(indice >=0 )
	{

		m_currentNurbs = m_listeNurbs[indice];
	}
}

QString NurbsManager::getCurrentName()
{
	return m_currentNurbs->getNameId();
}
QString NurbsManager::getUniqueName()
{
	QString res="Nurbs_1";

	bool trouver=false;
	int index = 1;
	while(!trouver)
	{

		bool temp = false;
		for(int i=0;i<m_listeNurbs.count();i++)
		{
			if( m_listeNurbs[i]->getNameId() == res)
			{
				temp=true;
			}
		}
		if( temp == true)
		{
			index++;
			res = "Nurbs_"+QString::number(index);
		}
		else
		{
			trouver = true;
		}
	}
	return res;
}

int NurbsManager::getIndex(QString name)
{
	for(int i=0;i<m_listeNurbs.count();i++)
	{
		if(m_listeNurbs[i]->getNameId()== name)
			return i;
	}

	return -1;
}

void NurbsManager::setPrecision(int value)
{
	m_precision = value;
	if(m_currentNurbs != nullptr)
	{
		m_currentNurbs->setPrecision(value);
	}
	/*for(int i=0;i<m_listeNurbs.count();i++)
	{
		m_listeNurbs[i]->setTriangulateResolution(m_precision);
	}*/
}

void NurbsManager::setInterpolation(bool value)
{
	m_interpolation = value;
	for(int i=0;i<m_listeNurbs.count();i++)
	{
		m_listeNurbs[i]->setLinearInterpolateInbetweens(m_interpolation);
	}
}

void NurbsManager::setWireframe(bool value)
{
	//m_wireframe = value;
	for(int i=0;i<m_listeNurbs.count();i++)
	{
		m_listeNurbs[i]->setWireframeRendering(value);
	}
}

void NurbsManager::loadNurbs(QString str,QVector3D posCam, Qt3DCore::QEntity* root,QObject* parent,IsoSurfaceBuffer surface ,QMatrix4x4 transform)
{
	QFileInfo fileinfo(str);


	/*Manager* nurbs = new Manager(m_workingset, fileinfo.baseName(),m_precision,posCam,root,surface, parent);
	m_workingset->addNurbs(nurbs->getNurbsData());
	nurbs->getNurbsData()->setDisplayPreference(true);*/

/*	std::vector<QVector3D> listepts;

	for(int i=0;i<listepoints.count()-1;i++)
	{
		listepts.push_back(listepoints[i]);
	}
*/

//	nurbs->createDirectriceFromTangent(path,transform);

	//nurbs->setDirectriceOk(listepoints.count());

///	connect(nurbs,SIGNAL(sendNurbsY(QVector3D,QVector3D)),this, SLOT(receiveNurbsY(QVector3D,QVector3D)));
//	connect(nurbs,SIGNAL(sendCurveDataTangent(QVector<PointCtrl>,bool ,QPointF)),this, SLOT(receiveCurveDataTangent(QVector<PointCtrl>,bool,QPointF)));
//	connect(nurbs,SIGNAL(sendCurveDataTangentOpt(GraphEditor_ListBezierPath*)),this, SLOT(receiveCurveDataTangentOpt(GraphEditor_ListBezierPath*)));
	//connect(nurbs,SIGNAL(sendCurveDataTangent2(QVector<QVector3D>,QVector<QVector3D>,bool ,QPointF)),this, SLOT(receiveCurveDataTangent2(QVector<QVector3D>,QVector<QVector3D>,bool,QPointF)));
//	connect(nurbs,SIGNAL(sendAnimationCam(int,QVector3D)),this, SLOT(setAnimationCamera(int, QVector3D)));

///	connect(nurbs,SIGNAL(generateDirectrice(QVector<PointCtrl>,QColor,bool)),this, SLOT(receiveNewGeneratrice(QVector<PointCtrl>,QColor,bool)));


	QStringList names =str.split("/");
	int nb = names.count();
	QString nameNurbs = names[nb-1];
	nameNurbs = nameNurbs.replace(".txt","");

	SelectNurbs(nameNurbs);
	if(m_currentNurbs !=nullptr)
	{
		m_currentNurbs->objToEditable(surface,posCam,root);
	}

	loadNurbsCurrent(fileinfo.baseName(),str,transform);

//	m_listeNurbs.push_back(nurbs);

	//m_currentNurbs= nurbs;
}


QColor NurbsManager::loadNurbsColor(QString path,int &layer)
{

	QColor colorNurbs(0,0,1);
	QFile file(path);
	if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		qDebug()<<"Manager Load nurbs ouverture du fichier impossible :"<<path;
		return colorNurbs;
	}


	bool trouver = false;


	QTextStream in(&file);
	while(!in.atEnd() && trouver ==false)
	{
		QString line = in.readLine();
		QStringList linesplit = line.split("|");
		if(linesplit.count()> 0)
		{
			if(linesplit[0] =="color")
			{
				QStringList color1 = linesplit[1].split("-");

				colorNurbs = QColor(color1[0].toInt(),color1[1].toInt(),color1[2].toInt());


			}
			else if(linesplit[0] =="precision")
			{

				if(linesplit.count()>= 3)layer = linesplit[2].toInt();
				trouver = true;
			}
		}
	}
	file.close();


	return colorNurbs;
}


void NurbsManager::loadNurbsCurrent(QString name, QString path,QMatrix4x4 transform)
{

	//TODO "NurbsManager::loadNurbsCurrent OBSOLETE";
	QFile file(path);
	if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		qDebug()<<"Manager Load nurbs ouverture du fichier impossible :"<<path;
		return;
	}


	QVector<PointCtrl> listeDirectrice;
	int widthTr;
	int heightTr;
	QColor colorDirectrice;
	QColor colorNurbs;
	QPolygonF polygonTr;
	int nbCurve;
	float  coef =-1.0f;
	int layer=-1;
	bool openGene=true;
	helperqt3d::IsectPlane plane;
	QVector<PointCtrl> listeGeneratrice;
	int widthAffine,heightAffine;
	QTextStream in(&file);

	int cptCurve =0;
	while(!in.atEnd())
	{
		QString line = in.readLine();
		QStringList linesplit = line.split("|");
		if(linesplit.count()> 0)
		{
			if(linesplit[0] =="color")
			{
				QStringList color1 = linesplit[1].split("-");
				QStringList color2 = linesplit[2].split("-");
				colorNurbs = QColor(color1[0].toInt(),color1[1].toInt(),color1[2].toInt());
				colorDirectrice = QColor(color2[0].toInt(),color2[1].toInt(),color2[2].toInt());
			}
			else if(linesplit[0] =="precision")
			{
				int precis = linesplit[1].toInt();
				if(linesplit.count() >=3) layer = linesplit[2].toInt();
			}
			else if(linesplit[0] =="directrice")
			{
				int nbptsDir = linesplit[1].toInt();
				int closed = linesplit[2].toInt();

				bool open = true;
				if(closed==1 ) open = false;


				for(int i=0;i<nbptsDir*6;i+=6)
				{
					QPointF pos(linesplit[i+3].toFloat(),linesplit[i+4].toFloat());
					QPointF ctrl1(linesplit[i+5].toFloat(),linesplit[i+6].toFloat());
					QPointF ctrl2(linesplit[i+7].toFloat(),linesplit[i+8].toFloat());

					listeDirectrice.push_back(PointCtrl(pos,ctrl1,ctrl2));

				}
				emit generateDirectrice(name,listeDirectrice,colorDirectrice,open);
			}
			else if(linesplit[0] =="nbcurve")
			{
				nbCurve =linesplit[1].toInt();
			}

			else if(linesplit[0] =="coef")
			{
				coef =linesplit[1].toFloat();
				if( coef <0.0f )coef = 0.0f;
			}
			else if(linesplit[0] =="plane")
			{
				plane.pointinplane =QVector3D (linesplit[1].toFloat(),linesplit[2].toFloat(),linesplit[3].toFloat());
				plane.xaxis = QVector3D(linesplit[4].toFloat(),linesplit[5].toFloat(),linesplit[6].toFloat());
				plane.yaxis = QVector3D(linesplit[7].toFloat(),linesplit[8].toFloat(),linesplit[9].toFloat());
			}
			else if(linesplit[0] =="nbpts")
			{

				listeGeneratrice.clear();
				int nbptsGene = linesplit[1].toInt();
				int closedG = linesplit[2].toInt();
				openGene = true;
				if(closedG==1 ) openGene = false;
				for(int i=0;i<nbptsGene*6;i+=6)
				{
					QPointF pos(linesplit[i+3].toFloat(),linesplit[i+4].toFloat());
					QPointF ctrl1(linesplit[i+5].toFloat(),linesplit[i+6].toFloat());
					QPointF ctrl2(linesplit[i+7].toFloat(),linesplit[i+8].toFloat());


					listeGeneratrice.push_back(PointCtrl(pos,ctrl1,ctrl2));
				}

			}
			else if(linesplit[0] =="randomTransformation")
			{

				widthTr = linesplit[1].toInt();
				heightTr = linesplit[2].toInt();
			}
			else if(linesplit[0] =="poly")
			{
				polygonTr.clear();
				int nbpoly = linesplit[1].toInt();
				for(int i=0;i<nbpoly*2;i+=2)
				{
					QPointF pos(linesplit[i+2].toFloat(),linesplit[i+3].toFloat());
					polygonTr.push_back(pos);

				}
			}
			else if(linesplit[0] =="affine")
			{
				widthAffine = linesplit[1].toInt();
				heightAffine = linesplit[2].toInt();
			}
			else if(linesplit[0] =="direct")
			{
				std::array<double,6> direct;
				direct[0] = linesplit[1].toDouble();
				direct[1] = linesplit[2].toDouble();
				direct[2] = linesplit[3].toDouble();
				direct[3] = linesplit[4].toDouble();
				direct[4] = linesplit[5].toDouble();
				direct[5] = linesplit[6].toDouble();

				Affine2DTransformation affine(widthAffine,heightAffine,direct);


				RandomTransformation* transfo = new RandomTransformation(heightTr,polygonTr,affine);

				//emit generateGeneratrice(listeGeneratrice,closeG);

				//)
				bool compute =false;

				if(cptCurve == nbCurve -1)compute=true;


				QPen pen;
				QBrush brush;
				GraphEditor_ListBezierPath* path = new GraphEditor_ListBezierPath(listeGeneratrice,pen,brush,nullptr,nullptr,!openGene);
				path->setNameNurbs(name);



				m_currentNurbs->setSliderXsectPos(coef,plane.pointinplane,-plane.getNormal());
				m_currentNurbs->addinbetweenXsectionClone(coef,false);
				m_currentNurbs->createGeneratriceFromTangent(path,transfo,compute,coef);

				cptCurve++;


			}
		}
	}
	file.close();

	m_currentNurbs->setColorNurbs(colorNurbs);
	m_currentNurbs->setTimeLayer(layer);






}

void NurbsManager::importNurbsObj(QString str,QString name, QVector3D posCam, Qt3DCore::QEntity* root,QObject* parent,bool refresh)
{



	int layer;
	QColor colorObj= loadNurbsColor(str,layer);


	QString pathObj  = str.replace(".txt",".obj");

	QFileInfo fileinfo(pathObj);
	if(fileinfo.exists()==false )return;

	Manager* nurbs = new Manager(m_workingset, name,posCam,root, parent);
	nurbs->setTimeLayer(layer);

	nurbs->importNurbsObj(pathObj,root,posCam,colorObj);

	connect(nurbs,SIGNAL(sendAnimationCam(int,QVector3D)),this, SLOT(setAnimationCamera(int, QVector3D)));


	m_workingset->addNurbs(nurbs->getNurbsData());
	nurbs->getNurbsData()->setAllDisplayPreference(true);

	m_listeNurbs.push_back(nurbs);

	m_currentNurbs= nurbs;

	//emit sendColorNurbs(m_currentNurbs->getColorDirectrice() ,m_currentNurbs->getColorNurbs(),m_currentNurbs->getPrecision(),m_currentNurbs->getModeEditable(),m_currentNurbs->getTimerLayer());

}


void NurbsManager::saveNurbs(QString str)
{

}

void NurbsManager::receiveNewGeneratrice(QVector<PointCtrl> listepoints ,QColor col,bool open)
{
	//emit generateDirectrice(m_currentNurbs->getNameId(),listepoints,col, open);
}





