#ifndef SeismicPropPanel_H
#define SeismicPropPanel_H

#include <QWidget>

class SeismicSurveyRep;

class SeismicSurveyPropPanel: public QWidget {
	Q_OBJECT
public:
	SeismicSurveyPropPanel(SeismicSurveyRep *rep, QWidget *parent = 0);
	virtual ~SeismicSurveyPropPanel();

private:
	SeismicSurveyRep *m_rep;
};

#endif
