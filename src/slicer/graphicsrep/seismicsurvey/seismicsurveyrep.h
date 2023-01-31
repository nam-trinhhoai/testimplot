#ifndef SeismicSurveyRep_H
#define SeismicSurveyRep_H

#include <QObject>
#include "abstractgraphicrep.h"
class SeismicSurvey;

class SeismicSurveyRep: public AbstractGraphicRep {
Q_OBJECT
public:
	SeismicSurveyRep(SeismicSurvey *survey, AbstractInnerView *parent = 0);
	virtual ~SeismicSurveyRep();

	//AbstractGraphicRep
	virtual bool canBeDisplayed() const override {
		return false;
	}

	virtual QWidget* propertyPanel() override;
	virtual GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;

	virtual void buildContextMenu(QMenu *menu) override;
	virtual IData* data() const override;
	virtual TypeRep getTypeGraphicRep() override;

private slots:
	void createRgbComposite();
	void AddSeismic();
	void openIsoHorizonInformation();

protected:
	SeismicSurvey *m_survey;


};

#endif
