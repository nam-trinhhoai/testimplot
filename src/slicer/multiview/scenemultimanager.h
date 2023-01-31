#ifndef SCENEMULTIMANAGER_H
#define SCENEMULTIMANAGER_H

#include <QObject>
#include <QVector2D>
#include <QVector3D>


namespace Qt3DCore {
class QTransform;
} // Qt3DCore



class SceneMultiManager : public QObject
{
    Q_OBJECT
  //  Q_PROPERTY(QVector3D dimensions READ dimensions NOTIFY dimensionsChanged)


public:
    explicit SceneMultiManager(QObject *parent = nullptr);
    ~SceneMultiManager();

    Q_INVOKABLE void addView();


private :

    //nombre maximum de fenetre de visualisation 3D
    int m_maxNbView3D;

    //nombre de fenetre de visualisation 3D courante
    int m_nbView3D;

};

#endif
