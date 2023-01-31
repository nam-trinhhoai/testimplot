#include "curvelistener.h"
#include "curvemodel.h"


CurveListener::CurveListener(QObject *parent) : QObject(parent)
{
}

void CurveListener::addCurve(std::shared_ptr<CurveModel> curvemodel)
{
    m_curves.push_back(curvemodel);
    connect(curvemodel.get(), &CurveModel::modelUpdated,      this, &CurveListener::curveUpdated);
    connect(curvemodel.get(), &CurveModel::modeltobeDeleted,  this, &CurveListener::curveDeleted);
    connect(curvemodel.get(), &CurveModel::modelSetSelected,  this, &CurveListener::curveSetSelected);

    //   qDebug() << " CurveListener size() " <<  m_curvestd::functions.size();
}

void CurveListener::removeCurve(CurveModel* curve)
{
    auto found=find_if(m_curves.begin(), m_curves.end(),    // find the index and delete it
                         [&] (std::shared_ptr<CurveModel> c) { return c.get() == curve; } );
    disconnect(curve, &CurveModel::modelUpdated,       this, &CurveListener::curveUpdated);
    disconnect(curve, &CurveModel::modeltobeDeleted,   this, &CurveListener::curveDeleted);
    disconnect(curve, &CurveModel::modelSetSelected,   this, &CurveListener::curveSetSelected);
    //   if position!= curves.end()
    m_curves.erase(found);
}

//  These two functions are attached to signals emitted from CurveModel via QT connect() mechanism
//  NB! User must have set m_curveUpdatedCallback and m_curveDeletedCallback first
void CurveListener::curveUpdated(bool finished, int index)
{
    //qDebug() <<m_muted<< " = mute, CurveListener::curveUpdated" << finished << " index " << index;
    if (m_muted) return;
    m_curveUpdatedCallback(finished, static_cast<CurveModel*>(sender()), index);
}

// Automatically called when <curve> object deletes itself
void CurveListener::curveDeleted()
{
    //   qDebug() << "CurveListener::curveDeleted" << m_curves.size();
    CurveModel* cm = static_cast<CurveModel*>(sender());
    removeCurve(cm);
    m_curveDeletedCallback();
}


void CurveListener::curveSetSelected(bool selected)
{
    CurveModel* cm = static_cast<CurveModel*>(sender());
    m_curveSelectedCallback(selected, cm);
}
//////////////////////////////



