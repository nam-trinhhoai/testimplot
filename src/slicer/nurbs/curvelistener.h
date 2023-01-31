#pragma once

#include <QObject>
#include <QDebug>
#include <functional>
class CurveModel;

//using namespace std;
// Listens to one or more curves and receives events from them
// Objects that want to receive such events should have an instance of a CurveListener and register callbacks
class CurveListener : public QObject
{
    Q_OBJECT
public:
    explicit CurveListener(QObject *parent = nullptr);

    typedef std::function<void(void)>                        FuncCurveDelete;
    typedef std::function<void(bool,CurveModel*, int index)> FuncCurveUpdate;
    typedef std::function<void(bool,CurveModel*)>            FuncCurveSelect;
    void setCallbackCurveDeleted( FuncCurveDelete curveDeleted) {m_curveDeletedCallback  = curveDeleted; };
    void setCallbackCurveUpdated( FuncCurveUpdate curveUpdated) {m_curveUpdatedCallback  = curveUpdated; };
    void setCallbackCurveSelected(FuncCurveSelect curveSelected){m_curveSelectedCallback = curveSelected;};

    void    addCurve(std::shared_ptr<CurveModel> curvemodel);
    void removeCurve(CurveModel* curve);

    void mute()  {m_muted=true; };
    void unmute(){m_muted=false;};

    const std::vector<  std::shared_ptr<CurveModel> >& getCurves() const {return m_curves;}

private:
    void curveUpdated(bool finished, int index);
    void curveDeleted();
    void curveSetSelected(bool selected);

    std::vector< std::shared_ptr<CurveModel> >  m_curves;
    bool m_muted = false;  // Don't call m_curveUpdatedCallback when curve changes

    // = is assigning temporary callback functions so code does not crash in case user forgets/does not need to use a certain setCallbackxx()
    FuncCurveDelete m_curveDeletedCallback    = []() {/*qDebug() << "setCallbackCurveDeleted unassigned";*/};
    FuncCurveUpdate m_curveUpdatedCallback    = [](bool val, CurveModel* cm, int index) {/*qDebug() << "setCallbackCurveUpdated unassigned";*/};
    FuncCurveSelect m_curveSelectedCallback   = [](bool val, CurveModel* cm) {/*qDebug() << "setCallbackCurveSelected " << cm << val;*/};
};
