package com.cardtable.platform

import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// The colours of the table screens: the theme's, read where they are drawn.
val Felt0 @Composable get() = LocalAppTheme.current.feltEdge
val Felt1 @Composable get() = LocalAppTheme.current.feltMid
val Felt2 @Composable get() = LocalAppTheme.current.feltCenter
val Ink @Composable get() = LocalAppTheme.current.ink
val InkDim @Composable get() = LocalAppTheme.current.inkDim
val Gold @Composable get() = LocalAppTheme.current.accent
val GoldBright @Composable get() = LocalAppTheme.current.accentBright
val CardRed = Color(0xFFC0282C)
val CardBlack = Color(0xFF1C1A16)
val CardBlue = Color(0xFF1D5BB0)
val CardGreen = Color(0xFF1E7B3C)
val Wood @Composable get() = LocalAppTheme.current.rim
val Panel @Composable get() = LocalAppTheme.current.panel

