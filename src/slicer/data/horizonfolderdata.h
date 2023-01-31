#ifndef HORIZONFOLDERDATA_H_
#define HORIZONFOLDERDATA_H_

#include <QObject>
#include <QList>
#include "folderdata.h"
#include "cudargbinterleavedimage.h"
#include "cpuimagepaletteholder.h"
#include "cubeseismicaddon.h"
#include "affinetransformation.h"
#include "freehorizon.h"

class HorizonFolderDataGraphicRepFactory;
class FixedRGBLayersFromDatasetAndCube;

struct LockerAttribut
{

	bool lock =false;
	QVector2D rangeR;
	QVector2D rangeG;
	QVector2D rangeB;
	bool modeRgb= false;

};

class HorizonFolderData : public IData {// FolderData {
	Q_OBJECT
public:
	HorizonFolderData(WorkingSetManager * workingSet,const QString &name, QObject* parent=0);

	HorizonFolderData(WorkingSetManager * workingSet,const QString &name,QStringList horizons, QObject* parent=0);
	~HorizonFolderData();

	//IData
	virtual IGraphicRepFactory *graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override{return m_name;}

/*	bool isDataContains(IData *data);
	void addData(IData *data);
	void removeData(IData *data);
	void deleteData(IData *data);
	QList<IData*> data();*/


	void select(int index);
	void deselect(int index);
	void changeOrder(int ,int);
	void setCurrentData(int );

	void moveData(int );


	void computeCache();
	void clearCache();
	void showCache(int);

	QStringList getAttributesAvailable();



	/*CUDARGBInterleavedImage* image()
	{
		return m_image;
	}
*/
	CPUImagePaletteHolder* isoSurfaceHolder()
	{

		if(m_currentLayer!= nullptr && m_currentLayer->m_attribut.size() >0 )
		{
			return m_currentLayer->m_attribut[0].getIsoSurface();
		}
		return nullptr;
	}


	int width()
	{
		if(m_currentLayer!= nullptr && m_currentLayer->m_attribut.size() >0 )
		{
			return m_currentLayer->m_attribut[0].width();
		}
		return 0;
	}

	int depth()
	{
		if(m_currentLayer!= nullptr && m_currentLayer->m_attribut.size() >0 )
		{
			return m_currentLayer->m_attribut[0].depth();
		}
		return 0;
	}

	int height()
	{
		if(m_currentLayer!= nullptr && m_currentLayer->m_attribut.size() >0 )
		{
			return m_currentLayer->m_attribut[0].heightFor3D();
		}
		return 0;
	}

	ImageFormats::QSampleType isoType()
	{
		ImageFormats::QSampleType val = ImageFormats::QSampleType::ERR;
		if(m_currentLayer!= nullptr && m_currentLayer->m_attribut.size() >0 )
		{
			return m_currentLayer->m_attribut[0].isoType();
		}
		return val;
	}




	const Affine2DTransformation* ijToXYTransfo() const
	{

		if(m_currentLayer!= nullptr && m_currentLayer->m_attribut.size() >0 )
		{
			return m_currentLayer->m_attribut[0].ijToXYTransfo();
		}
		return nullptr;
	}

	const Affine2DTransformation* ijToInlineXlineTransfoForInline() const
		{

			if(m_currentLayer!= nullptr && m_currentLayer->m_attribut.size() >0 )
			{
				return m_currentLayer->m_attribut[0].ijToInlineXlineTransfoForInline();
			}
			return nullptr;
		}

	const Affine2DTransformation* ijToInlineXlineTransfoForXline() const
			{

				if(m_currentLayer!= nullptr && m_currentLayer->m_attribut.size() >0 )
				{
					return m_currentLayer->m_attribut[0].ijToInlineXlineTransfoForInline();
				}
				return nullptr;
			}

	const QColor getHorizonColor() const
	{
		if(m_currentLayer!= nullptr && m_currentLayer->m_attribut.size() >0 )
		{
			return m_currentLayer->color();
		}
		return Qt::white;
	}


	const AffineTransformation* sampleTransformation() const
	{
		if(m_currentLayer!= nullptr)
			return m_currentLayer->sampleTransformation();

		return nullptr;
	}

	CubeSeismicAddon cubeSeismicAddon()
	{
		if(m_currentLayer!= nullptr)
			return m_currentLayer->cubeSeismicAddon();
		return CubeSeismicAddon();
	}

	QList<FreeHorizon*> completOrderList()
	{
		return m_completOrderList;
	}

	FreeHorizon* currentLayer()
	{
		return m_currentLayer;
	}



	FreeHorizon::Attribut currentLayerWithAttribut(QString attribut)
	{
		FreeHorizon::Attribut emptyAttribut;
		if(m_currentLayer == nullptr)return emptyAttribut;
		return m_currentLayer->getLayer(attribut);//m_attributNane
	}

	QString getPathSave(int index)
	{

		if( m_completOrderList.size() >0 && index <m_completOrderList.size())
		{
			//qDebug()<< " --> getPathSave "<< m_completOrderList[0]->path();
			return m_completOrderList[index]->path();
		}
		//qDebug()<< " --> m_completOrderList.size() "<< m_completOrderList.size();
		return "";
	}

	int getOrderList(int index)
	{
		if( m_OrderList.size() >0 && index <m_OrderList.size())
		{
			//qDebug()<< " --> getPathSave "<< m_completOrderList[0]->path();
			return m_OrderList[index];
		}
		//qDebug()<< " --> m_completOrderList.size() "<< m_completOrderList.size();
		return -1;
	}

	int getNbFreeHorizon()
	{
		return m_completOrderList.size();
	}



	// range lock
	bool isRangeLocked(QString nameAttribut) const;
	const QVector2D& lockedRangeRed(QString nameAttribut) const;
	const QVector2D& lockedRangeGreen(QString nameAttribut) const;
	const QVector2D& lockedRangeBlue(QString nameAttribut) const;
	void lockRange(const QVector2D& rangered,const QVector2D& rangegreen,const QVector2D& rangeblue,QString nameAttribut,bool mode);
	void unlockRange(QString nameAttribut);




/*	void setNameAttribut(QString nom)
	{
		m_nameAttribut= nom;
	}*/
signals:
void rangeLockChanged();
	void layerAdded(IData *data);
	void layerRemoved(IData *data);

	void orderChanged(int index, int newindex);

	void currentChanged(/*FixedRGBLayersFromDatasetAndCube* m_currentLayer*/);

	void requestComputeCache();
	void requestClearCache();
	void requestShowCache(int);

public slots:

	void added(IData* data);
	void removed(IData* data);
	//void deleted(IData* data);

private:
	void changeUp(int , int );
	void changeDown(int , int );
	void changeTop(int);
	void changeBottom(int);

//	QString m_nameAttribut="";

	QList<FreeHorizon*> m_completOrderList;
	QList<int> m_OrderList;

	FreeHorizon* m_currentLayer = nullptr;
	std::map<QString ,LockerAttribut> m_lockerAtt;



	QString m_name;
	QUuid m_uuid;


//	CUDARGBInterleavedImage* m_image = nullptr;
//	CPUImagePaletteHolder* m_isoSurfaceHolder= nullptr;


	HorizonFolderDataGraphicRepFactory * m_repFactory;
};

Q_DECLARE_METATYPE(HorizonFolderData*)

#endif
