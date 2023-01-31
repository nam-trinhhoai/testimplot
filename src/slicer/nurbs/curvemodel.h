#pragma once

#include <vector>
#include <QVector3D>
#include "QObject"

class CurveModel : public QObject
{
    Q_OBJECT

signals:
    void modelUpdated(bool finished, int index=-1);
    void modeltobeDeleted(CurveModel* cm);
    void modelSetSelected(bool selected);


public:
    CurveModel(){}
    CurveModel(const CurveModel &c){m_vecPositions = c.m_vecPositions;};
    ~CurveModel(){};

    unsigned int getSize() const { return static_cast<uint>(m_vecPositions.size());}
    QVector3D& getPosition(unsigned int index){return m_vecPositions[index];}

    void setPoint(uint id, const QVector3D& p)
    {
        if (id>m_vecPositions.size()) return;
        m_vecPositions[id] = p;
        emit modelUpdated(false, id);
    }

    void setAllPoints(QVector<QVector3D> liste)
    {
    	clear();
    	for(int i=0;i<liste.size();i++)
    	{
    		m_vecPositions.push_back(liste[i]);
    		//emit modelUpdated(false,i);
    	}
    	emitModelUpdated(true);
    }

    // CurveController/Curveinstantiator listens to the signals from curvemodel
    // Notice how currently, only InsertBack emits a modelUpdated event
    // For other cases, the user must manually take care of calling emitModelUpdated/emitModeltobeDeleted.
    void clear(){m_vecPositions.clear(); }
    void eraseAt(unsigned int index){m_vecPositions.erase(m_vecPositions.begin()+index);}
    void insertAfter(unsigned int index, QVector3D& element){m_vecPositions.insert(m_vecPositions.begin()+index,element);}
    void insertBack(QVector3D pos){m_vecPositions.push_back(pos);  emit modelUpdated(false);}
    void insertBackSilent(QVector3D pos){m_vecPositions.push_back(pos);}

    void eraseBack(unsigned int uIndex){m_vecPositions.erase(m_vecPositions.begin()+uIndex,m_vecPositions.end()); }

    void emitModelUpdated(bool val,int index=-1){emit modelUpdated(val, index); }
    void emitModeltobeDeleted(){emit modeltobeDeleted(this); } // emitted when object has been decided to be deleted
    void emitModelSetSelected(bool selected){emit modelSetSelected(selected);}

    const std::vector<QVector3D>& data() const { return m_vecPositions; }

private:
    std::vector<QVector3D> m_vecPositions;
};


