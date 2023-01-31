/*
 * NurbsManager.h
 *
 *  Created on: 23 fevrier 2022
 *      Author: l1049100 (sylvain)
 */

#ifndef NURBSMANAGER_H
#define NURBSMANAGER_H

#include <QVector>
#include <QVector3D>
#include <Qt3DCore/QEntity>
#include <QObject>
#include <QColor>
#include "workingsetmanager.h"

#include "manager.h"

#include "PointCtrl.h"

class RandomTransformation;
class Affine2DTransformation;

class NurbsManager: public QObject
{
    Q_OBJECT
public:

	explicit NurbsManager(QObject* parent);

	virtual ~NurbsManager();

	void setColorNurbs(QColor col);
	void setColorNurbs(QString nameNurbs, QColor col);


	void setColorDirectrice(QColor col);
	void setPositionLight(QVector3D pos);

	void addNurbsFromTangent(QVector3D posCam,Qt3DCore::QEntity* root,QObject* parent, QVector<QVector3D> listepoints,IsoSurfaceBuffer surface,QString nameNurb,GraphEditor_ListBezierPath* path ,QMatrix4x4 transform,QColor col);
	void addNurbs(QVector3D posCam,Qt3DCore::QEntity* root,QObject* parent, QVector<QVector3D> listepoints,IsoSurfaceBuffer surface);
	void deleteNurbs(int index);
	void deleteNurbs(QString s);
	void deleteCurrentNurbs();

	void deleteCurrentGeneratrice(QString);
	void destroyNurbs(QString name);

	void addXSection();

	void updateDirectriceWithTangent(GraphEditor_ListBezierPath* path,QMatrix4x4 transform,QColor col);
	void updateDirectrice(QVector<QVector3D> listepoints);
	void updateGeneratrice(QVector<QVector3D> listepoints, int index);

	void addGeneratriceFromTangent(QVector<QVector3D> listepoints, int index,bool isopen=true);
	void addGeneratrice(QVector<QVector3D> listepoints, int index,bool isopen=true);
	void addGeneratrice(QVector<PointCtrl> listeCtrls,QVector<QVector3D> listepoints, int index,QVector<QVector3D>  listeCtrl3D,QVector<QVector3D>  listeTangente3D,QVector3D cross3d,bool isopen=true, QPointF cross = QPointF(0,0));

	void addGeneratrice(GraphEditor_ListBezierPath* path,RandomTransformation* transfo);
	void setPrecision(int);
	void setInterpolation(bool);
	void setWireframe(bool);

	void SelectNurbs(int index);
	void SelectNurbs(QString );

	void setSliderXsection(float pos);
	QVector3D setSliderXsection(float pos,QVector3D position, QVector3D normal);
	void createNewXSection(float pos);
	void createNewXSectionClone(float pos);

	QString getCurrentName();

	Manager* exists(QString name);

	QVector<Manager*> getListeNurbs()
	{
		return m_listeNurbs;
	}

	void setWorkingSetManager(WorkingSetManager* workingset)
	{
		m_workingset = workingset;
	}

	void setCurrentRandom(RandomView3D* rand);

	RandomView3D* getCurrentRandom();
	RandomView3D* getRandom(QString name);


	Manager* getCurrentNurbs(){
		return m_currentNurbs;
	}

	Manager* getNurbsFromRandom(RandomView3D*);

	float getSliderXSection();

	void loadNurbs(QString str,QVector3D posCam, Qt3DCore::QEntity* root,QObject* parent,IsoSurfaceBuffer surface ,QMatrix4x4 transform);
	void loadNurbsCurrent(QString name, QString path,QMatrix4x4 transform);
	QColor loadNurbsColor(QString path,int& layer);
	void saveNurbs(QString);
	void createNurbsSimple( QString nameNurbs,IsoSurfaceBuffer buffer,Qt3DCore::QEntity* root,QObject* parent);
	void exportNurbsObj(QString);
	void importNurbsObj(QString str,QString name, QVector3D posCam, Qt3DCore::QEntity* root,QObject* parent,bool refresh=false);

	signals:
		void sendCurveData(std::vector<QVector3D>,bool);
		void sendCurveDataTangent(QVector<PointCtrl> listepoints,bool isopen,QPointF cross);
		void sendCurveDataTangent2(QVector<QVector3D> listepoints3D,QVector<QVector3D> globalTangente3D,bool isopen,QPointF cross, QString nameNurbs);
		void sendCurveDataTangentOpt(GraphEditor_ListBezierPath* path);
		void generateDirectrice(QString, QVector<PointCtrl>,QColor,bool);

		void sendAnimationCam(int, QVector3D );
		void sendPosition(QVector3D);
		void sendNurbsY(QVector3D,QVector3D);
		void sendColorNurbs(QColor ,QColor,int,bool,int);

	public slots:
		void receiveCurveData(std::vector<QVector3D>,bool);
		void receiveCurveDataTangent(QVector<PointCtrl> listepoints,bool isopen,QPointF);
		void receiveCurveDataTangent2(QVector<QVector3D> listepoints3D,QVector<QVector3D> globalTangente3D,bool isopen,QPointF cross);
		void receiveCurveDataTangentOpt(GraphEditor_ListBezierPath* path);
		void setAnimationCamera(int ,QVector3D);
		void receiveNurbsY(QVector3D,QVector3D);
		void receiveNewGeneratrice(QVector<PointCtrl> listepoints ,QColor col,bool open);

private:

	QString getUniqueName();
	int getLayerTime();
	int getIndex(QString name);


	QVector<Manager*> m_listeNurbs;

	WorkingSetManager* m_workingset= nullptr;

	int m_precision = 40;
	bool m_interpolation = false;

	QPointer<Manager> m_currentNurbs = nullptr;


};


#endif
