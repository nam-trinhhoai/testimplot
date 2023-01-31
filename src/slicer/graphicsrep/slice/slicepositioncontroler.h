#ifndef SlicePositionControler_h
#define SlicePositionControler_h

#include <QObject>
#include <QColor>

#include "datacontroler.h"
#include "sliceutils.h"

class SliceRep;
class SlicePositionControler : public DataControler{
	  Q_OBJECT
public:
    SlicePositionControler(SliceRep *rep,QObject *parent);
	virtual ~SlicePositionControler();

	SliceDirection direction() const;
	void setDirection(SliceDirection dir);

	int position()const;

	QColor color() const;
	void  setColor(const QColor &c);

	virtual QUuid dataID() const override;
public	slots:
	void requestPosChanged(int val);
	void setPositionFromRep(int val);
signals:
	void posChanged(int);
private:
	SliceDirection m_dir;

	int m_position;
	QColor m_color;

	SliceRep* m_rep;
};


#endif
