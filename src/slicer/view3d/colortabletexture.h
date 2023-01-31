#ifndef COLORTABLETEXTURE_H
#define COLORTABLETEXTURE_H

#include <Qt3DRender/QTexture>
#include "lookuptable.h"

class ColorTableTexture : public Qt3DRender::QTexture1D
{
    Q_OBJECT
    Q_PROPERTY(LookupTable lookupTable READ lookupTable WRITE setLookupTable NOTIFY lookupTableChanged)
public:
    explicit ColorTableTexture(Qt3DCore::QNode *parent = nullptr);
    ~ColorTableTexture();

    void setLookupTable(const LookupTable &table);
    LookupTable lookupTable() const;

signals:
    void lookupTableChanged();

private:
    LookupTable m_lookupTable;
};

#endif // COLORTABLETEXTURE_H
