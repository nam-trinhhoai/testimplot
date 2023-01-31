#include "horizonfolderdata.h"

#include <QMessageBox>

#include <cmath>
#include <iostream>
#include "gdal.h"
#include <algorithm>
#include "horizonfolderdatagraphicrepfactory.h"
#include "fixedrgblayersfromdatasetandcube.h"

HorizonFolderData::HorizonFolderData(WorkingSetManager *workingSet, const QString &name,
		QObject *parent) :
		IData(workingSet,parent) {
	m_name = name;
	m_uuid = QUuid::createUuid();

	connect(workingSet->folders().horizonsFree, SIGNAL(dataRemoved(IData*)),this ,SLOT(removed(IData*)));

	QList<IData*> listdata = workingSet->folders().horizonsFree->data();
	for(int i=0;i<listdata.size();i++)
	{
		FreeHorizon* fixed = dynamic_cast<FreeHorizon*>(listdata[i]);
		if(fixed != nullptr)
		{
			m_completOrderList.push_back(fixed);
			m_OrderList.push_back(i);
		}
	}

//	m_image= nullptr;
//	m_isoSurfaceHolder = nullptr;

	m_repFactory = new HorizonFolderDataGraphicRepFactory(this);
}


HorizonFolderData::HorizonFolderData(WorkingSetManager *workingSet, const QString &name,QStringList horizons, QObject *parent) :
		IData(workingSet,parent) {
	m_name = name;
	m_uuid = QUuid::createUuid();

	connect(workingSet->folders().horizonsFree, SIGNAL(dataRemoved(IData*)),this ,SLOT(removed(IData*)));


	QStringList horizonName;
	for(int i=0;i<horizons.size();i++)
	{
		QFileInfo info(horizons[i]);
		QString basename = info.baseName();
		horizonName<<basename;

	}

	QList<IData*> listdata = workingSet->folders().horizonsFree->data();
	int counter = 0;
	for(int i=0;i<horizonName.size();i++)
	{
		bool notFound = true;
		int j = 0;
		while (notFound && j<listdata.size())
		{
			FreeHorizon* fixed = dynamic_cast<FreeHorizon*>(listdata[j]);
			notFound = fixed==nullptr || horizonName[i].compare(fixed->name())!=0;
			if(!notFound)
			{
				m_completOrderList.push_back(fixed);
				m_OrderList.push_back(counter);
				counter++;
			}
			j++;
		}
	}

//	m_image= nullptr;
//	m_isoSurfaceHolder = nullptr;

	m_repFactory = new HorizonFolderDataGraphicRepFactory(this);
}


IGraphicRepFactory* HorizonFolderData::graphicRepFactory() {
	return m_repFactory;
}


HorizonFolderData::~HorizonFolderData() {

}


QUuid HorizonFolderData::dataID() const {
	return m_uuid;
}


QStringList HorizonFolderData::getAttributesAvailable()
{
	QList<FreeHorizon*> listFree =completOrderList();
	QStringList listAttributs;


	for(int i=1;i<listFree.count();i++)
	{
		for(int j=0;j< listFree[i]->m_attribut.size();j++)
		{
			QString name = listFree[i]->m_attribut[j].name();
			if(name!="" && ! listAttributs.contains(name))
			{

				listAttributs.append(name);
			}
		}
	}
	return listAttributs;
}


bool HorizonFolderData::isRangeLocked(QString nameAttribut) const {
	auto ite = m_lockerAtt.find(nameAttribut);
	if(ite != m_lockerAtt.end())
	{
		return ite->second.lock;
	}

	return false;
}

const QVector2D& HorizonFolderData::lockedRangeRed(QString nameAttribut) const
{
	auto ite = m_lockerAtt.find(nameAttribut);
	if(ite != m_lockerAtt.end())
	{
		return ite->second.rangeR;
	}
	return QVector2D(0,0);
}

const QVector2D& HorizonFolderData::lockedRangeGreen(QString nameAttribut) const
{
	auto ite = m_lockerAtt.find(nameAttribut);
	if(ite != m_lockerAtt.end())
	{
		return ite->second.rangeG;
	}
	return QVector2D(0,0);
}

const QVector2D& HorizonFolderData::lockedRangeBlue(QString nameAttribut) const
{
	auto ite = m_lockerAtt.find(nameAttribut);
	if(ite != m_lockerAtt.end())
	{
		return ite->second.rangeB;
	}
	return QVector2D(0,0);
}

void HorizonFolderData::lockRange(const QVector2D& rangered,const QVector2D& rangegreen,const QVector2D& rangeblue,QString nameAttribut,bool mode)
{
	auto ite = m_lockerAtt.find(nameAttribut);
	if(ite != m_lockerAtt.end())
	{
		if( (ite->second.rangeR !=rangered) || (ite->second.rangeG !=rangegreen) || (ite->second.rangeB !=rangeblue) ||  ite->second.lock ==false &&
				rangered.x() < rangered.y() && rangegreen.x() <rangegreen.y() && rangeblue.x() < rangeblue.y() )
			{
				ite->second.rangeR =rangered;
				ite->second.rangeG =rangegreen;
				ite->second.rangeB =rangeblue;

				ite->second.lock = true;
				emit rangeLockChanged();
			}
	}
	else
	{
		if(rangered.x() < rangered.y() && rangegreen.x() <rangegreen.y() && rangeblue.x() < rangeblue.y() )
		{
			LockerAttribut locker;

			locker.lock = true;
			locker.rangeR= rangered;
			locker.rangeG = rangegreen;
			locker.rangeB = rangeblue;
			locker.modeRgb= mode;
			m_lockerAtt[nameAttribut] = locker;
		}
	}
}

void HorizonFolderData::unlockRange(QString nameAttribut)
{
	auto ite = m_lockerAtt.find(nameAttribut);
	if(ite != m_lockerAtt.end())
	{
		if (ite->second.lock==true )
		{

			ite->second.lock = false;
			emit rangeLockChanged();
		}
	}

}


void HorizonFolderData::setCurrentData(int indexCompletList)
{
	int id =m_OrderList.indexOf(indexCompletList);
	if(id>=0)
	{

		m_currentLayer = m_completOrderList[indexCompletList];

		emit currentChanged();

	}
	else
	{
		m_currentLayer = nullptr;
	}
}


void HorizonFolderData::moveData(int indexCompletList)
{
//timer
//	setCurrentData(indexCompletList);
}


void HorizonFolderData::added(IData* data)
{
	FreeHorizon* fixed = dynamic_cast<FreeHorizon*>(data);
	if(fixed != nullptr)
	{
		m_completOrderList.push_back(fixed);
		m_OrderList.push_back(m_completOrderList.count()-1);

		emit layerAdded(data);
	}
}

void HorizonFolderData::removed(IData* data)
{
	int index =m_completOrderList.indexOf(data);
	if (index<0)
	{
		return; // removed data not in list
	}
	m_completOrderList.removeOne(data);
	int orderIndex = m_OrderList[index];
	m_OrderList.removeOne(index);

	// update order
	int nextMin = -1; // if data need to be changed, find the highest layer lower than orderIndex
	int nextMax = m_OrderList.size(); // if data need to be changed, find the lowest layer greater than orderIndex
	for (int i=0; i<m_OrderList.size(); i++)
	{
		if (m_OrderList[i]>orderIndex)
		{
			m_OrderList[i]--;
		}
		if (m_OrderList[i]<=orderIndex && m_OrderList[i]>nextMin)
		{
			nextMin = m_OrderList[i];
		}
		if (m_OrderList[i]>=orderIndex && m_OrderList[i]<nextMax)
		{
			nextMax = m_OrderList[i];
		}
	}
	// change current layer if needed
	if (data==m_currentLayer)
	{
		if (nextMax<m_OrderList.size())
		{
			setCurrentData(nextMax);
		}
		else if (nextMin>=0)
		{
			setCurrentData(nextMin);
		}
		else
		{
			setCurrentData(-1);
		}
	}

	emit layerRemoved(data);

	QMessageBox::information(nullptr, tr("Horizon removed from animation"), tr("Horizon : ") + data->name() + tr(" has been removed from animation : ") + name());
}

void HorizonFolderData::changeOrder(int indexComplet, int newIndexComplet)
{

	if(indexComplet<newIndexComplet)changeUp(indexComplet,newIndexComplet);
	else if(indexComplet > newIndexComplet)changeDown(indexComplet,newIndexComplet);

	//if(newIndexComplet == 0 && indexComplet > 0)changeTop(indexComplet);
	//if(newIndexComplet == m_completOrderList.count()-1  && indexComplet <  m_completOrderList.count()-2) changeBottom(indexComplet);
}

void HorizonFolderData::changeUp(int indexComplet, int newIndexComplet)
{
	//qDebug()<<"avant m_OrderList = "<<m_OrderList;
	m_completOrderList.move(indexComplet,newIndexComplet);

	auto iter = std::lower_bound(m_OrderList.begin(),m_OrderList.end(),indexComplet);

	m_OrderList.erase(iter);

	while (iter!= m_OrderList.end() && (*iter)<= newIndexComplet)
	{
		(*iter)--;

		iter++;
	}

	m_OrderList.insert(iter,newIndexComplet);

	//qDebug()<<" m_OrderList = "<<m_OrderList;
	emit orderChanged(indexComplet,newIndexComplet);


}

void HorizonFolderData::changeDown(int indexComplet, int newIndexComplet)
{
	//qDebug()<<"avant m_OrderList = "<<m_OrderList;
	//qDebug()<<"avant m_OrderList = "<<m_OrderList;

	//for(int i=0;i<3;i++ )qDebug()<<i<<" -completlist:"<<m_completOrderList[i]->name();
	m_completOrderList.move(indexComplet,newIndexComplet);
	//for(int i=0;i<3;i++ )qDebug()<<i<<" +completlist:"<<m_completOrderList[i]->name();
	auto iter = std::lower_bound(m_OrderList.begin(),m_OrderList.end(),indexComplet);
	if(iter== m_OrderList.begin())return;

	m_OrderList.erase(iter);
	//qDebug()<<" m_OrderList = "<<m_OrderList;
	iter--;
	while (iter!= m_OrderList.begin() && (*iter)>= newIndexComplet)
	{
		if(iter!= m_OrderList.end())(*iter)++;
		iter--;
	}
	if(iter== m_OrderList.begin() && (*iter)>= newIndexComplet)
		(*iter)++;
	else if(m_OrderList.size()>0) iter++;
	m_OrderList.insert(iter,newIndexComplet);

	//qDebug()<<" m_OrderList = "<<m_OrderList;
	emit orderChanged(indexComplet,newIndexComplet);
}


void HorizonFolderData::changeTop(int indexComplet)
{
	int newIndexComplet = 0;
	m_completOrderList.move(indexComplet,newIndexComplet);

		auto iter = std::lower_bound(m_OrderList.begin(),m_OrderList.end(),indexComplet);

		m_OrderList.erase(iter);

		while (iter!= m_OrderList.end() && (*iter)<= newIndexComplet)
		{
			(*iter)--;

			iter++;
		}

		m_OrderList.insert(iter,indexComplet);

		//qDebug()<<" m_OrderList = "<<m_OrderList;
		emit orderChanged(indexComplet,newIndexComplet);
}


void HorizonFolderData::changeBottom(int indexComplet)
{
	int newIndexComplet = m_completOrderList.count();
	m_completOrderList.move(indexComplet,newIndexComplet);

	auto iter = std::lower_bound(m_OrderList.begin(),m_OrderList.end(),indexComplet);

	m_OrderList.erase(iter);

	while (iter!= m_OrderList.begin() && (*iter)>= newIndexComplet)
	{
		(*iter)++;
		iter--;
	}
	if(m_OrderList.size()>0) iter++;
	m_OrderList.insert(iter,indexComplet);

	//qDebug()<<" m_OrderList = "<<m_OrderList;
	emit orderChanged(indexComplet,newIndexComplet);
}


void HorizonFolderData::select(int indexCompleteList)
{
	auto iter = std::lower_bound(m_OrderList.begin(),m_OrderList.end(),indexCompleteList);
	m_OrderList.insert(iter,indexCompleteList);

}

void HorizonFolderData::deselect(int indexCompleteList)
{
	m_OrderList.removeOne(indexCompleteList);
}

void HorizonFolderData::computeCache()
{
	emit requestComputeCache();
}
void HorizonFolderData::clearCache()
{
	emit requestClearCache();
}

void HorizonFolderData::showCache(int i)
{
	emit requestShowCache(i);
}
